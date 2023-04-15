import os
import sys
import numpy as np
from data.data import ModelData
import tensorflow as tf
from input_output.tech.arg_parser import ArgumentParser
from input_output.tech.saver import save_attack_results

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'



if __name__ == "__main__":
    parser = ArgumentParser()
    model = parser.get_model()
    model_data = ModelData(dataset=parser.get_dataset_name(), model=model,
                           attack=parser.get_attack(), adv_present=parser.get_adv_data())
    model_data.split_data(val_split=parser.get_val_split())
    model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.train(batch_size=parser.get_batches(), epochs=parser.get_epochs(),
                train_data=(model_data.X_train, model_data.y_train),
                validation_data=(model_data.X_valid, model_data.y_valid),
                weights=parser.get_weights(),
                dataset=parser.get_dataset_name())
    # model.evaluate(data=model_data.X_test, labels=model_data.y_test, title="CLEAN")
    model_data.generate_adversarial_data(model=model, attack=parser.get_attack(),
                                         dataset=parser.get_dataset_name(),
                                         adv_present=parser.get_adv_data())
    model.evaluate(data=model_data.X_adv, labels=model_data.y_test, title="ADV")
    save_attack_results(model=model, data=model_data, attack=parser.get_attack(),
                        dataset=parser.get_dataset_name(), save=parser.get_save())






