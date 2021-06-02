from handwriting_synthesis import training
from handwriting_synthesis import utils


if __name__ == '__main__':
    max_length = 50
    dataset_factory = utils.DataSetFactory('../iam_ondb_home', num_examples=20, max_length=max_length)
    train_set, val_set = dataset_factory.split_data()
    train_task = training.HandwritingPredictionTrainingTask()
    loop = training.TrainingLoop(train_set, batch_size=4, training_task=train_task)

    cb = training.HandwritingGenerationCallback(train_task._model, 'samples', max_length, train_set, iteration_interval=1)

    loop.add_callback(cb)
    loop.start(100)
