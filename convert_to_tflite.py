import tensorflow as tf

def main():
    model = tf.keras.models.load_model('trained_model.tf')
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open('trained_model.tflite', 'wb') as f:
        f.write(tflite_model)


if __name__ == '__main__':
    main()
