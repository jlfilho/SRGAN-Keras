# encoding: utf-8

import os
import sys
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
warnings.filterwarnings("ignore")
#stderr = sys.stderr
#sys.stderr = open(os.devnull, 'w')

import datetime
import tensorflow as tf
import numpy as np
import restore

from keras.models import Model
from keras.layers import Input, Add, BatchNormalization
from keras.layers import LeakyReLU, Conv2D, Dense, PReLU, Lambda
from keras.optimizers import Adam
#from keras.utils.data_utils import OrderedEnqueuer, SequenceEnqueuer, GeneratorEnqueuer
from tensorflow.keras.utils import OrderedEnqueuer, GeneratorEnqueuer, SequenceEnqueuer
from keras.callbacks import TensorBoard, ModelCheckpoint, LambdaCallback
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, LearningRateScheduler
from keras import backend as K
from tqdm import tqdm

from util import DataLoader, plot_test_images 

from losses import psnr3 as psnr
from losses import VGGLoss



class SRGAN():
    """ 
    Implementation of SRGAN as described in the paper:
    Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network
    https://arxiv.org/abs/1609.04802
    """

    def __init__(self, 
        height_lr=24, width_lr=24, channels=3,
        upscaling_factor=4, 
        gen_lr=1e-4, dis_lr=1e-4, loss_weights=[0.006, 1e-4], 
        training_mode=True,
        colorspace = 'RGB'
    ):
                 
        """        
        :param int height_lr: Height of low-resolution images
        :param int width_lr: Width of low-resolution images
        :param int channels: Image channels
        :param int upscaling_factor: Up-scaling factor
        :param int gen_lr: Learning rate of generator
        :param int dis_lr: Learning rate of discriminator
        """
        
        
        # Low-resolution image dimensions
        self.height_lr = height_lr
        self.width_lr = width_lr
        self.training_mode = training_mode

        # High-resolution image dimensions
        if upscaling_factor not in [2, 4, 8]:
            raise ValueError('Upscaling factor must be either 2, 4, or 8. You chose {}'.format(upscaling_factor))
        self.upscaling_factor = upscaling_factor
        self.height_hr = int(self.height_lr * self.upscaling_factor)
        self.width_hr = int(self.width_lr * self.upscaling_factor)

        # Low-resolution and high-resolution shapes
        self.channels = channels
        self.colorspace = colorspace
        self.shape_lr = (self.height_lr, self.width_lr, self.channels)
        self.shape_hr = (self.height_hr, self.width_hr, self.channels)

        # Learning rates
        self.gen_lr = gen_lr
        self.dis_lr = dis_lr
        
        # Gan setup settings
        self.loss_weights=loss_weights
        self.VGGLoss = VGGLoss(self.shape_hr)
        self.gen_loss =  'mse' 
        self.content_loss = self.VGGLoss.content_loss # self.VGGLoss.euclidean_content_loss
        self.adversarial_loss = 'binary_crossentropy'
        
        # Build & compile the generator network
        self.generator = self.build_generator()
        self.compile_generator(self.generator)

        # If training, build rest of GAN network
        if training_mode:
            self.discriminator = self.build_discriminator()
            self.compile_discriminator(self.discriminator)
            self.srgan = self.build_srgan()
            self.compile_srgan(self.srgan)


    def save_weights(self, filepath):
        """Save the generator and discriminator networks"""
        self.generator.save_weights("{}_generator_{}X.h5".format(filepath, self.upscaling_factor))
        self.discriminator.save_weights("{}_discriminator_{}X.h5".format(filepath, self.upscaling_factor))

    def load_weights(self, generator_weights=None, discriminator_weights=None, **kwargs):
        print(">> Loading weights...")
        if generator_weights:
            self.generator.load_weights(generator_weights, **kwargs)
        if discriminator_weights:
            self.discriminator.load_weights(discriminator_weights, **kwargs)
            
    def SubpixelConv2D(self, name, scale=2):
        """
        Keras layer to do subpixel convolution.
        NOTE: Tensorflow backend only. Uses tf.depth_to_space
        
        :param scale: upsampling scale compared to input_shape. Default=2
        :return:
        """

        def subpixel_shape(input_shape):
            dims = [input_shape[0],
                    None if input_shape[1] is None else input_shape[1] * scale,
                    None if input_shape[2] is None else input_shape[2] * scale,
                    int(input_shape[3] / (scale ** 2))]
            output_shape = tuple(dims)
            return output_shape

        def subpixel(x):
            return tf.depth_to_space(x, scale)

        return Lambda(subpixel, output_shape=subpixel_shape, name=name)


    def build_generator(self, residual_blocks=16):
        """
        Build the generator network according to description in the paper.

        :param optimizer: Keras optimizer to use for network
        :param int residual_blocks: How many residual blocks to use
        :return: the compiled model
        """


        def residual_block(input):
            x = Conv2D(64, kernel_size=3, strides=1, padding='same')(input)
            if self.training_mode:
                x = BatchNormalization(momentum=0.8)(x)
            x = PReLU(shared_axes=[1,2])(x)            
            x = Conv2D(64, kernel_size=3, strides=1, padding='same')(x)
            if self.training_mode:
                x = BatchNormalization(momentum=0.8)(x)
            x = Add()([x, input])
            return x

        def upsample(x, number):
            x = Conv2D(256, kernel_size=3, strides=1, padding='same', name='upSample_Conv2d_'+str(number))(x)
            x = self.SubpixelConv2D('upSample_SubPixel_'+str(number), 2)(x)
            x = PReLU(shared_axes=[1,2], name='upSample_PReLU_'+str(number))(x)
            return x

        # Input low resolution image
        lr_input = Input(shape=(None, None, self.channels),name='Input-gen')

        # Pre-residual
        x_start = Conv2D(64, kernel_size=9, strides=1, padding='same',name='Conv2d-pre')(lr_input)
        x_start = PReLU(shared_axes=[1,2],name='PReLU-pre')(x_start)

        # Residual blocks
        x = residual_block(x_start)
        for _ in range(residual_blocks - 1):
            x = residual_block(x)


        # Post-residual block
        x = Conv2D(64, kernel_size=3, strides=1, padding='same',name='Conv-pos')(x)
        if self.training_mode:
            x = BatchNormalization(momentum=0.8,name='BN-pos')(x)
        x = Add()([x, x_start])
        
        # Upsampling depending on factor
        x = upsample(x, 1)
        if self.upscaling_factor > 2:
            x = upsample(x, 2)
        if self.upscaling_factor > 4:
            x = upsample(x, 3)

        
        # Generate high resolution output
        # tanh activation, see: 
        # https://towardsdatascience.com/gan-ways-to-improve-gan-performance-acf37f9f59b
        x = Conv2D(
            self.channels, 
            kernel_size=9, 
            strides=1, 
            padding='same', 
            activation='tanh',name='Conv-out'
        )(x)

        # Create model 
        model = Model(inputs=lr_input, outputs=x,name='Generator')        
        #model.summary()
        return model
  
    def build_discriminator(self, filters=64):
        """
        Build the discriminator network according to description in the paper.

        :param optimizer: Keras optimizer to use for network
        :param int filters: How many filters to use in first conv layer
        :return: the compiled model
        """

        def conv2d_block(input, filters, strides=1, bn=True):
            d = Conv2D(filters, kernel_size=3, strides=strides, padding='same')(input)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            d = LeakyReLU(alpha=0.2)(d)
            return d

        # Input high resolution image
        img = Input(shape=self.shape_hr)
        x = conv2d_block(img, filters, bn=False)
        
        x = conv2d_block(x, filters, strides=2)
        x = conv2d_block(x, filters*2)
        x = conv2d_block(x, filters*2, strides=2)
        x = conv2d_block(x, filters*4)
        x = conv2d_block(x, filters*4, strides=2)
        x = conv2d_block(x, filters*8)
        x = conv2d_block(x, filters*8, strides=2)
        x = Dense(filters*16)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dense(1, activation='sigmoid')(x)

        # Create model and compile
        model = Model(inputs=img, outputs=x,name='Discriminator')
        #model.summary()
        return model

  
    def build_srgan(self):
        """Create the combined SRGAN network"""

        # Input LR images
        img_lr = Input(self.shape_lr,name='input_gan')

        # Create a high resolution image from the low resolution one
        generated_hr = self.generator(img_lr)

        # In the combined model we only train the generator
        self.discriminator.trainable = False

        # Determine whether the generator HR images are OK
        generated_check = self.discriminator(generated_hr)

        # Create sensible names for outputs in logs
        generated_hr = Lambda(lambda x: x, name='Content')(generated_hr)
        generated_check = Lambda(lambda x: x, name='Adversarial')(generated_check)

        # Create model and compile
        # Using binary_crossentropy with reversed label, to get proper loss, see:
        # https://danieltakeshi.github.io/2017/03/05/understanding-generative-adversarial-networks/
        model = Model(inputs=img_lr, outputs=[generated_hr,generated_check],name='GAN')
        #model.summary()        
        return model




    def compile_generator(self, model):
        """Compile the generator with appropriate optimizer"""
        model.compile(
            loss=self.gen_loss,
            optimizer=Adam(lr=self.gen_lr, beta_1=0.9),
            metrics=[psnr]
        )

    def compile_discriminator(self, model):
        """Compile the generator with appropriate optimizer"""
        model.compile(
            loss=self.adversarial_loss,
            optimizer=Adam(lr=self.dis_lr, beta_1=0.9),
            metrics=['accuracy']
        )


    def compile_srgan(self, model):
        """Compile the GAN with appropriate optimizer"""
        model.compile(
            loss=[self.content_loss,self.adversarial_loss],
            loss_weights=self.loss_weights,
            optimizer=Adam(lr=self.gen_lr, beta_1=0.9)
        )

    def train_generator(self,
        epochs=None, batch_size=None,
        workers=None,
        max_queue_size=None,
        modelname=None, 
        datapath_train=None,
        datapath_validation='../',
        datapath_test='../',
        steps_per_epoch=None,
        steps_per_validation=None,
        crops_per_image=None,
        print_frequency=None,
        log_weight_path='./model/', 
        log_tensorboard_path='./logs/',
        log_tensorboard_update_freq=None,
        log_test_path="./test/",
        media_type='i'
    ):
        """Trains the generator part of the network with MSE loss"""


        # Create data loaders
        train_loader = DataLoader(
            datapath_train, batch_size,
            self.height_hr, self.width_hr,
            self.upscaling_factor,
            crops_per_image,
            media_type,
            self.channels,
            self.colorspace
        )

        
        validation_loader = None 
        if datapath_validation is not None:
            validation_loader = DataLoader(
                datapath_validation, batch_size,
                self.height_hr, self.width_hr,
                self.upscaling_factor,
                crops_per_image,
                media_type,
                self.channels,
                self.colorspace
        )

        test_loader = None
        if datapath_test is not None:
            test_loader = DataLoader(
                datapath_test, 1,
                self.height_hr, self.width_hr,
                self.upscaling_factor,
                1,
                media_type,
                self.channels,
                self.colorspace
        )

        
        # Callback: tensorboard
        callbacks = []
        if log_tensorboard_path:
            tensorboard = TensorBoard(
                log_dir=os.path.join(log_tensorboard_path, modelname),
                histogram_freq=0,
                batch_size=batch_size,
                write_graph=True,
                write_grads=True,
                update_freq=log_tensorboard_update_freq
            )
            callbacks.append(tensorboard)
        else:
            print(">> Not logging to tensorboard since no log_tensorboard_path is set")

	# Callback: Stop training when a monitored quantity has stopped improving
        earlystopping = EarlyStopping(
            monitor='val_loss', 
	        patience=500, verbose=1, 
	        restore_best_weights=True     
        )
        callbacks.append(earlystopping)
        
        # Callback: save weights after each epoch
        modelcheckpoint = ModelCheckpoint(
            os.path.join(log_weight_path, modelname + '_{}X.h5'.format(self.upscaling_factor)), 
            monitor='val_loss', 
            save_best_only=True, 
            save_weights_only=True
        )
        callbacks.append(modelcheckpoint)

        # Callback: Reduce lr when a monitored quantity has stopped improving
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                    patience=50, min_lr=1e-5,verbose=1)
        callbacks.append(reduce_lr)

        # Learning rate scheduler
        def lr_scheduler(epoch, lr):
            factor = 0.5
            decay_step = 100 #100 epochs * 2000 step per epoch = 2x1e5
            if epoch % decay_step == 0 and epoch:
                return lr * factor
            return lr
        lr_scheduler = LearningRateScheduler(lr_scheduler, verbose=1)
        callbacks.append(lr_scheduler)
 
        
         # Callback: test images plotting
        if datapath_test is not None:
            testplotting = LambdaCallback(
                on_epoch_end=lambda epoch, logs: None if ((epoch+1) % print_frequency != 0 ) else plot_test_images(
                    self.generator,
                    test_loader,
                    datapath_test,
                    log_test_path,
                    epoch+1,
                    name=modelname,
                    channels=self.channels,
                    colorspace=self.colorspace))
        callbacks.append(testplotting)

        # Use several workers on CPU for preparing batches
        enqueuer = OrderedEnqueuer(
            train_loader,
            use_multiprocessing=True
        )
        enqueuer.start(workers=workers, max_queue_size=max_queue_size)
        output_generator = enqueuer.get()

                            
        # Fit the model
        self.generator.fit_generator(
            output_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=validation_loader,
            validation_steps=steps_per_validation,
            callbacks=callbacks,
            use_multiprocessing=False, #workers>1 because single gpu
            workers=workers
        )

    def train_srgan(self, 
        epochs=None, batch_size=16, 
        modelname=None, 
        datapath_train=None,
        datapath_validation=None, 
        steps_per_validation=1000,
        datapath_test=None, 
        workers=4, max_queue_size=10,
        first_epoch=0,
        print_frequency=1,
        crops_per_image=2,
        log_weight_frequency=None, 
        log_weight_path='./model/', 
        log_tensorboard_path='./data/logs/',
        log_tensorboard_update_freq=10,
        log_test_frequency=500,
        log_test_path="./images/samples/", 
        media_type='i'        
    ):
        """Train the SRGAN network

        :param int epochs: how many epochs to train the network for
        :param str modelname: name to use for storing model weights etc.
        :param str datapath_train: path for the image files to use for training
        :param str datapath_test: path for the image files to use for testing / plotting
        :param int print_frequency: how often (in epochs) to print progress to terminal. Warning: will run validation inference!
        :param int log_weight_frequency: how often (in epochs) should network weights be saved. None for never
        :param int log_weight_path: where should network weights be saved        
        :param int log_test_frequency: how often (in epochs) should testing & validation be performed
        :param str log_test_path: where should test results be saved
        :param str log_tensorboard_path: where should tensorflow logs be sent
        """

        
        
         # Create data loaders
        train_loader = DataLoader(
            datapath_train, batch_size,
            self.height_hr, self.width_hr,
            self.upscaling_factor,
            crops_per_image,
            media_type,
            self.channels,
            self.colorspace
        )

        # Validation data loader
        validation_loader = None 
        if datapath_validation is not None:
            validation_loader = DataLoader(
                datapath_validation, batch_size,
                self.height_hr, self.width_hr,
                self.upscaling_factor,
                crops_per_image,
                media_type,
                self.channels,
                self.colorspace
        )

        test_loader = None
        if datapath_test is not None:
            test_loader = DataLoader(
                datapath_test, 1,
                self.height_hr, self.width_hr,
                self.upscaling_factor,
                1,
                media_type,
                self.channels,
                self.colorspace
        )
    
        # Use several workers on CPU for preparing batches
        enqueuer = OrderedEnqueuer(
            train_loader,
            use_multiprocessing=True,
            shuffle=True
        )
        enqueuer.start(workers=workers, max_queue_size=max_queue_size)
        output_generator = enqueuer.get()
        
        # Callback: tensorboard
        if log_tensorboard_path:
            tensorboard = TensorBoard(
                log_dir=os.path.join(log_tensorboard_path, modelname),
                histogram_freq=0,
                batch_size=batch_size,
                write_graph=False,
                write_grads=False,
                update_freq=log_tensorboard_update_freq
            )
            tensorboard.set_model(self.srgan)
        else:
            print(">> Not logging to tensorboard since no log_tensorboard_path is set")
        
        # Callback: format input value
        def named_logs(model, logs):
            """Transform train_on_batch return value to dict expected by on_batch_end callback"""
            result = {}
            for l in zip(model.metrics_names, logs):
                result[l[0]] = l[1]
            return result

        # Shape of output from discriminator
        disciminator_output_shape = list(self.discriminator.output_shape)
        disciminator_output_shape[0] = batch_size
        disciminator_output_shape = tuple(disciminator_output_shape)

        # VALID / FAKE targets for discriminator
        real = np.ones(disciminator_output_shape)
        fake = np.zeros(disciminator_output_shape) 
               

        # Each epoch == "update iteration" as defined in the paper        
        print_losses = {"GAN": [], "D": []}
        start_epoch = datetime.datetime.now()
        
        # Random images to go through
        idxs = np.random.randint(0, len(train_loader), epochs)        
        
        # Loop through epochs / iterations
        for epoch in range(first_epoch, int(epochs)+first_epoch):

            # Start epoch time
            if epoch % (print_frequency + 1) == 0:
                start_epoch = datetime.datetime.now()            

            # Train discriminator 
            self.discriminator.trainable = True
            #real = np.ones(disciminator_output_shape) - np.random.random_sample(disciminator_output_shape)*0.05
            #fake = np.random.random_sample(disciminator_output_shape)*0.05  
            labels = np.concatenate([real, fake])
            imgs_lr, imgs_hr = next(output_generator)
            generated_hr = self.generator.predict(imgs_lr)
            combined_images = np.concatenate([imgs_hr, generated_hr])
            discriminator_loss = self.discriminator.train_on_batch(combined_images, labels)
            #real_loss = self.discriminator.train_on_batch(imgs_hr, real)
            #print("Real: ",real_loss)
            #fake_loss = self.discriminator.train_on_batch(generated_hr, fake)
            #print("Fake: ",fake_loss)
            #discriminator_loss = 0.5 * np.add(real_loss, fake_loss)
            

            # Train generator
            self.discriminator.trainable = False
            #real = np.ones(disciminator_output_shape) - np.random.random_sample(disciminator_output_shape)*0.05  
            #imgs_lr, imgs_hr = next(output_generator)
            #gan_loss = self.srgan.train_on_batch(imgs_lr, [imgs_hr,real])

            """ real = np.ones(disciminator_output_shape) - np.random.random_sample(disciminator_output_shape)*0.2 """  
            
            #for _ in tqdm(range(1),ncols=1,desc=">> Training generator:"):
            imgs_lr, imgs_hr = next(output_generator)
            gan_loss = self.srgan.train_on_batch(imgs_lr, [imgs_hr,real])

     
            # Callbacks
            logs = named_logs(self.srgan, gan_loss)
            tensorboard.on_epoch_end(epoch, logs)

            # Save losses            
            print_losses['GAN'].append(gan_loss)
            print_losses['D'].append(discriminator_loss)

            # Show the progress
            if epoch % print_frequency == 0:
                g_avg_loss = np.array(print_losses['GAN']).mean(axis=0)
                d_avg_loss = np.array(print_losses['D']).mean(axis=0)
                print("\nEpoch {}/{} | Time: {}s\n>> GAN: {}\n>> Discriminator: {}".format(
                    epoch, epochs+first_epoch,
                    (datetime.datetime.now() - start_epoch).seconds,
                    ", ".join(["{}={:.4f}".format(k, v) for k, v in zip(self.srgan.metrics_names, g_avg_loss)]),
                    ", ".join(["{}={:.4f}".format(k, v) for k, v in zip(self.discriminator.metrics_names, d_avg_loss)])
                ))
                print_losses = {"GAN": [], "D": []}

                # Run validation inference if specified
                if datapath_validation:
                    validation_losses = self.generator.evaluate_generator(
                        validation_loader,
                        steps=steps_per_validation,
                        use_multiprocessing=workers>1,
                        workers=workers
                    )
                    print(">> Validation Losses: {}".format(
                        ", ".join(["{}={:.4f}".format(k, v) for k, v in zip(self.generator.metrics_names, validation_losses)])
                    ))                

            # If test images are supplied, run model on them and save to log_test_path
            if datapath_test and epoch % log_test_frequency == 0:
                plot_test_images(self.generator, test_loader, datapath_test, log_test_path, epoch, modelname,
                channels = self.channels,colorspace=self.colorspace)

            # Check if we should save the network weights
            if log_weight_frequency and epoch % log_weight_frequency == 0:
                # Save the network weights
                self.save_weights(os.path.join(log_weight_path, modelname))

    def predict(self,
            lr_path = None,
            sr_path = None,
            print_frequency = False,
            qp = 8,
            fps = None,
            media_type = None,
            gpu=False 
        ):
        """ lr_videopath: path of video in low resoluiton
            sr_videopath: path to output video 
            print_frequency: print frequncy the time per frame and estimated time, if False no print 
            crf: [0,51] QP parameter 0 is the best quality and 51 is the worst one
            fps: framerate if None is use the same framerate of the LR video
            media_type: type of media 'v' to video and 'i' to image
        """
        if(media_type == 'v'):
            time_elapsed = restore.write_srvideo(self.generator,lr_path,sr_path,self.upscaling_factor,print_frequency=print_frequency,crf=qp,fps=fps,gpu=gpu)
        elif(media_type == 'i'):
            time_elapsed = restore.write_sr_images(self.generator, lr_imagepath=lr_path, sr_imagepath=sr_path,scale=self.upscaling_factor)
        else:
            print(">> Media type not defined or not suported!")
            return 0
        return time_elapsed

# Run the SRGAN network
if __name__ == "__main__":

    
    """ 
    # --------------------------------------------------------------------------------------
    print(">> Creating the SRResNet network")
    SRResNet = SRGAN(upscaling_factor=2,channels=3,colorspace='RGB',training_mode=True)
    SRResNet.load_weights('../model/SRResNet_places365_2X.h5')
    i=1

    datapath = '../../data/videoset/1080p/' 
    outpath = '../out/SRResNet/540p_2X/' 
    for dirpath, _, filenames in os.walk(datapath):
        for filename in [f for f in sorted(filenames) if any(filetype in f.lower() for filetype in ['jpeg', 'png', 'jpg','mp4','264','webm','wma'])]:
            print(os.path.join(dirpath, filename),outpath+filename.split('.')[0]+'.mp4')
            if(i>0):
                t = SRResNet.predict(
                        lr_path=os.path.join(dirpath, filename), 
                        sr_path=outpath+filename.split('.')[0]+'.mp4',
                        qp=0,
                        media_type='v',
                        gpu=True
                    )
            i+=1 

    datapath = '../../data/videoset/720p/' 
    outpath = '../out/SRResNet/360p_2X/'
    for dirpath, _, filenames in os.walk(datapath):
        for filename in [f for f in sorted(filenames) if any(filetype in f.lower() for filetype in ['jpeg', 'png', 'jpg','mp4','264','webm','wma'])]:
            print(os.path.join(dirpath, filename),outpath+filename.split('.')[0]+'.mp4')
            if i > 48:
                t = SRResNet.predict(
                        lr_path=os.path.join(dirpath, filename), 
                        sr_path=outpath+filename.split('.')[0]+'.mp4',
                        qp=0,
                        media_type='v',
                        gpu=True
                    )
            i+=1"""
    
    # Instantiate the SRGAN object
    print(">> Creating the SRGAN network")
    SRResNet = SRGAN(upscaling_factor=2,channels=3,colorspace='RGB',training_mode=True)
    SRResNet.load_weights('../model/SRGAN_places365_generator_2X.h5')
    i=1
    datapath = '../../data/videoset/1080p/' 
    outpath = '../out/SRGAN/540p_2X/' 
    for dirpath, _, filenames in os.walk(datapath):
        for filename in [f for f in sorted(filenames) if any(filetype in f.lower() for filetype in ['jpeg', 'png', 'jpg','mp4','264','webm','wma'])]:
            print(os.path.join(dirpath, filename),outpath+filename.split('.')[0]+'.mp4')
            if i > 144:
                t = SRResNet.predict(
                        lr_path=os.path.join(dirpath, filename), 
                        sr_path=outpath+filename.split('.')[0]+'.mp4',
                        qp=0,
                        media_type='v',
                        gpu=True
                    )
            i+=1

    datapath = '../../data/videoset/720p/' 
    outpath = '../out/SRGAN/360p_2X/'
    for dirpath, _, filenames in os.walk(datapath):
        for filename in [f for f in sorted(filenames) if any(filetype in f.lower() for filetype in ['jpeg', 'png', 'jpg','mp4','264','webm','wma'])]:
            print(os.path.join(dirpath, filename),outpath+filename.split('.')[0]+'.mp4')
            t = SRResNet.predict(
                    lr_path=os.path.join(dirpath, filename), 
                    sr_path=outpath+filename.split('.')[0]+'.mp4',
                    qp=0,
                    media_type='v',
                    gpu=True
                ) 



    # Train the SRGAN
    """ gan.train_srgan(
        epochs=1000,
        batch_size=16,
        modelname='SRGAN',
        datapath_train='../../../data/train_large/',
        datapath_validation='../../data/val_large/',        
        steps_per_validation=10,
        datapath_test='../../data/benchmarks/Set5/', 
        workers=4, max_queue_size=10,
        first_epoch=0,
        print_frequency=1,
        crops_per_image=2,
        log_weight_frequency=2, 
        log_weight_path='../model/', 
        log_tensorboard_path='../logs/',
        log_tensorboard_update_freq=10,
        log_test_frequency=10,
        log_test_path="../test/"
    ) """

    """ gan.train_generator(
        epochs=50,batch_size=16,workers=1,
        modelname='SRResNet',
	    datapath_train='../../../data/train_large/',
	    datapath_validation='../../data/val_large/',
	    datapath_test='../../data/benchmarks/Set5/',
	    steps_per_epoch=10,
        print_frequency=1,
        steps_per_validation=10,
        crops_per_image=4,
        log_weight_path='../model/', 
        log_tensorboard_path='../logs/',
        log_tensorboard_update_freq=10,
        log_test_path="../test/"
    ) """
