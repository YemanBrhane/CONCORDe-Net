import tensorflow as tf
import tensorflow.keras.backend as K
# from GenericUnet import GetGenericUnet


class GetDiceCellCountModel(object):
    def __init__(self,
                 depth,
                 input_shape,
                 optimizer_type='adam',
                 cell_counter_model_dir=None,
                 pretrained=True,
                 backend='vgg',
                 cell_count_base_neurons=16,
                 learning_rate=0.001,
                 base=16,
                 loss_type='dice_and_count',
                 epochs=400):
        self. optimizer_type = optimizer_type
        self.backend = backend
        self.cell_count_base_neurons = cell_count_base_neurons
        self.learning_rate = learning_rate
        self.base = base
        self.depth = depth
        self.epochs = epochs
        self.input_shape = input_shape
        self.pretrained = pretrained
        self.loss_type = loss_type
        self.cell_counter_model_dir = cell_counter_model_dir

    @staticmethod
    def inception_block(input_tensor, num_filters):
    
        p1 = tf.keras.layers.Conv2D(num_filters, (1, 1), padding='same', activation='relu')(input_tensor)
        p1 = tf.keras.layers.Conv2D(num_filters, (3, 3), padding='same', activation='relu')(p1)
    
        p2 = tf.keras.layers.Conv2D(num_filters, (3, 3), padding='same', activation='relu')(input_tensor)
        p2 = tf.keras.layers.Conv2D(num_filters, (3, 3), padding='same', activation='relu')(p2)
    
        p3 = tf.keras.layers.Conv2D(num_filters, (1, 1), padding='same', activation='relu')(input_tensor)
    
        # return tf.keras.layers.concatenate([p1, p2, p3], axis=3)
        o = tf.keras.layers.Add()([p1, p2, p3])
    
        return o

    def get_inception_backend_unet(self, input_tensor):
        # input_shape = (self.patch_size, self.patch_size, self.CH)
        en = self.inception_block(input_tensor, self.base)
        en = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))(en)
        features = self.base
        for i in range(2, self.depth):
            features = 2 * features
            en = self.inception_block(en, features)
            en = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))(en)
    
        features = 2 * features
    
        en = self.inception_block(en, features)
        encoder_model = tf.keras.Model(inputs=[input_tensor], outputs=[en])
    
        # DECODER
        features = int(features / 2)
        d = tf.keras.layers.concatenate([tf.keras.layers.Conv2DTranspose(features, (2, 2), strides=(2, 2),
                                                                         padding='same')(
            encoder_model.layers[-1].output), encoder_model.layers[-8].output], axis=3)
        d = self.inception_block(d, features)
    
        a = 2
        for j in range(2, self.depth):
            features = int(features / 2)
            d = tf.keras.layers.concatenate([tf.keras.layers.Conv2DTranspose(features, (2, 2), strides=(2, 2),
                                                                             padding='same')(d),
                                             encoder_model.layers[-7 * a - 1].output], axis=3)
            d = self.inception_block(d, features)
        
            a += 1
    
        d = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(d)
    
        # unet_model = tf.keras.Model(inputs=[input_tensor], outputs=[d])
        #
        # unet_model.summary()
        # # tf.keras.utils.plot_model(unet_model, 'model.png', show_shapes=True)
        # print('number of layers:{}'.format(len(unet_model.layers)))
        # print('depth:{}'.format(self.depth))
    
        return d

    def get_vgg_backend_unet(self, input_tensor):
    
        return 0

    # def get_model(self):
    #     pass
    
    @staticmethod
    def dice_coef(y_true, y_pred):
        smooth = 1
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        
        return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    def dice_coef_loss(self, y_true, y_pred):
        return 1-self.dice_coef(y_true, y_pred)

    @staticmethod
    def count_loss(y_true, y_pred):
        #    y_pred = K.round(y_pred)
        smooth = 1
        mean_abs_diff = K.mean(K.abs(y_pred - y_true))
        return 1 - 1 / (smooth + mean_abs_diff)  # 1- 1/K.exp(sum_square_diff)

    def count_number_of_cells(self, input_tensor):
        if self.pretrained is True and self.cell_counter_model_dir is not None:
            print('loading pretrained counter model')
            counter_model = tf.keras.models.load_model(self.cell_counter_model_dir, custom_objects={'count_loss': self.count_loss})
            # plot_model(counter_model, 'model_counter.png')
            # counter_model.summary()
            for layer in counter_model.layers:
                layer.trainable = False

            return counter_model(input_tensor)
        else:
            print('creating counter model')
            X = tf.keras.layers.Conv2D(self.cell_count_base_neurons, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='count_conv1')(input_tensor)
            X = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(X)

            X = tf.keras.layers.Conv2D(self.cell_count_base_neurons * 2, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='count_conv2')(X)
            X = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(X)

            X = tf.keras.layers.Conv2D(self.cell_count_base_neurons * 4, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', name='count_conv3')(X)
            X = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(X)

            X = tf.keras.layers.Flatten()(X)
            X = tf.keras.layers.Dense(200, activation='relu', name='count_dense1')(X)
            X = tf.keras.layers.Dropout(rate=0.3)(X)
            cell_count = tf.keras.layers.Dense(1, activation='relu', name='cell_count')(X)

            return cell_count

    def get_cell_counter_pretrained(self, input_tensor):
        counter_model = tf.keras.models.load_model(self.cell_counter_model_dir, custom_objects={'count_loss': self.count_loss})
        tf.keras.utils.plot_model(counter_model, 'model_counter.png')
        counter_model.summary()
        for layer in counter_model.layers:
            layer.trainable = False

        return counter_model(input_tensor)

    def get_model(self):
        # obj = GetGenericUnet(base=self.base, backend=self.backend)
        # unet_output = obj.get_unet_model_not_build(input_tensor)
        input_tensor = tf.keras.layers.Input(shape=self.input_shape)

        if self.backend == 'vgg':
            print('backend= vgg')
            unet_output = self.get_vgg_backend_unet(input_tensor)

        elif self.backend == 'inception':
            print('backend= inception')
            unet_output = self.get_inception_backend_unet(input_tensor)
        else:
            raise Exception('Unknown backend, self.backend= {}'.format(self.backend))

        # return tf.keras.models.Model(inputs=[input_tensor], outputs=[unet_output])

        if self.loss_type == 'dice':
            model = tf.keras.models.Model(inputs=[input_tensor], outputs=unet_output)

            if self.optimizer_type == 'adam':
                opt_ = tf.keras.optimizers.Adam(lr=self.learning_rate)
                model.compile(optimizer=opt_, loss=self.dice_coef_loss)
            else:
                decay = self.learning_rate / self.epochs
                opt_ = tf.keras.optimizers.SGD(lr=self.learning_rate, momentum=0.9, decay=decay, nesterov=False)
                model.compile(optimizer=opt_, loss=self.dice_coef_loss)
        else:
            # seg_output = IdentityLayer(name='seg_output')(unet_output)
            seg_output = tf.keras.layers.Lambda(lambda x: x, name='seg_output')(unet_output)
            cell_count_output = self.count_number_of_cells(input_tensor=unet_output)

            model = tf.keras.Model(inputs=[input_tensor], outputs=[seg_output, cell_count_output])

            if self.optimizer_type == 'adam':
                opt_ = tf.keras.optimizers.Adam(lr=self.learning_rate)
                model.compile(optimizer=opt_,
                              loss=[self.dice_coef_loss, self.count_loss],
                              loss_weights=[1, 0.3])
            else:
                decay = self.learning_rate/self.epochs
                opt_ = tf.keras.optimizers.SGD(lr=self.learning_rate, momentum=0.9, decay=decay, nesterov=False)
                model.compile(optimizer=opt_,
                              loss=[self.dice_coef_loss, self.count_loss],
                              loss_weights=[1, 0.3])
        return model

if __name__ == '__main__':
    X = r'PretrainedCellCounter\bestweights.h5'
    obj = GetDiceCellCountModel(cell_counter_model_dir=X,
                                pretrained=True,
                                backend='inception',
                                depth=5,
                                input_shape=(224, 224, 3))
    M = obj.get_model()
    # tf.keras.utils.plot_model(M, 'model.png')
    M.summary()