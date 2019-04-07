import functools
import pickle


def test_on_folds(model):
    with open('folds.pkl', 'rb') as f:
        folds = pickle.load(f)

    accuracy = 0
    for train, test in zip(
            [folds[:i] + folds[i + 1:] for i in range(len(folds))],
            folds
    ):
        train = functools.reduce(lambda x, y: x + y, train)
        model.train(train)
        accuracy += model.predict_test(test)
        print(accuracy)

    print(f'Score: {accuracy / 5}')


def test_on_test_train(model):
    with open('folds.pkl', 'rb') as f:
        folds = pickle.load(f)

    accuracy = 0
    for train, test in zip(
            [folds[:i] + folds[i + 1:] for i in range(len(folds))],
            folds
    ):
        train = functools.reduce(lambda x, y: x + y, train)
        model.train(train)
        accuracy += model.predict_test(test)
        print(accuracy)

    print(f'Score: {accuracy / 5}')
