---
layout: post
title:  "Federated Farming, Part 2 - Models"
date:   2019-07-17 00:16:00 +0100
categories: jekyll update
---

These are my notes on implementing the federated learning approach presented in [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629v3). You can find all the code [here](https://github.com/distracteddreamer/fedfarm). All mistakes are my own. 

# Introduction

Here we implement and train some federated learning models using the data partitions we created in [part 1](/papertrail/jekyll/update/2019/07/17/Federated-Farming-Data.html).  All the code for the models is in the file [Federated_Farming-Model.py](https://github.com/distracteddreamer/fedfarm/blob/master/Federated_Farming-Model.py). For convenience I am using Keras to build the models. 

First we create a dataloader for Keras based on this helpful [tutorial](https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly). It has the option of passing in a `client_colm` which refers to the type of shard ("shard_iid" or "shard_non_iid") and a `num` argument which is the shard id, so we can have a separate data generator for each client.


```python
class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, df_path, 
                 batch_size, 
                 img_size, 
                 n_classes,
                 client_colm=None,
                 num=None, 
                 shuffle=True):
        'Initialization'
        self.num = num
        self.shuffle = shuffle
        df = pd.read_csv(df_path)
        if num is not None:
            rows = df[df[client_colm]==self.num]
        else:
            rows = df
        self.batch_size = batch_size
        self.img_size = img_size
        self.n_classes = n_classes
        self.filenames = rows.filename.values
        self.labels = rows.label.values
        
        self.on_epoch_end()
```

We use a simple ConvNet architecture based on the one introducted in the paper.


```python
def simple_cnn(img_size, n_classes):
        
    inputs = Input(img_size)
    conv1 = Conv2D(kernel_size=5, filters=32)(inputs)
    pool1 = MaxPooling2D(pool_size=2, strides=2)(conv1)
    conv2 = Conv2D(kernel_size=5, filters=64)(pool1)
    pool2 = MaxPooling2D(pool_size=2, strides=2)(conv2)
    flat = Flatten()(pool2)
    dense = Dense(units=512, activation='relu')(flat)
    out = Dense(units=n_classes, activation='softmax')(dense)
    model = Model(inputs=inputs, outputs=out)
    
    return model
```

The evaluation metric for the competition is [accuracy](https://www.kaggle.com/c/plant-seedlings-classification/discussion/46728#latest-400410) which is not the best metric for evaluating on unbalanced data but for consistency we will keep it for the experiments and also because although the distribution of classes is not uniform the proportions of different classes don’t wildly vary either.


```python
def get_model(config):
    model = simple_cnn(config.data.img_size, config.data.n_classes)
    model.compile(optimizer=SGD(lr=config.train.learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['acc'])
    return model
```

# Baseline
As a baseline we will train it on the full train dataset. The code for this is in the function `train_basic`. Our configuration is defined as EasyDicts from the easydict module (`pip install easydict`) since, as the name implies, it is convenient to access attributes, particularly nested attributes. 

We won’t balance the batches or make much effort to optimise the model performance since as they note in the paper with regard to their CIFAR10 experiment "our goal is to evaluate our optimization method, not achieve the best possible accuracy on this task". This model converges to a little over 60% accuracy, starting to overfit about a third of the way through.

![png]({{ site.baseurl }}/assets/Federated_Farming-Model/output_7_0.png)

![png]({{ site.baseurl }}/assets/Federated_Farming-Model/output_8_0.png)

## Federated averaging
I will go through the most important parts of the implementation here. The key ideas are as follows:

- There are several **rounds** of training 
- A global model is initialised
- During each round of training local models are trained on a small number of randomly chosen clients with a given **batch size** and **number of epochs**.
- The global model is updated with the weighted average of the weights from the local models of the chosen clients.

We implement the following algorithm from the paper in the function ```fed_averaging```:

![png]({{ site.baseurl }}/assets/Federated_Farming-Model/fed_algorithm.png)


The core of ```fed_averaging``` is as follows:


```python
for t in range(1, config.train.num_rounds + 1):
    print('Round {}'.format(t))
    print('-' * 10)
    print('Training')
    global_weights = model.get_weights()
    _global_weights = [i.copy() for i in global_weights]
    m = int(np.ceil(max(config.train.client_fraction * config.train.num_clients, 1)))
    clients = np.random.permutation(config.train.num_clients)[:m]
    local_results = []

    for i, client in enumerate(clients):
        model.set_weights(global_weights)
        results = client_update(config, client, model)
        local_results.append(results)


    local_weights, n_examples, _tloss, _tacc = zip(*local_results)
    tloss = np.mean(_tloss)
    tacc = np.mean(_tacc)
    model.set_weights(average_weights(local_weights, n_examples))
```

Since our goal is to understand how the model performs when trained in this manner with different partitions of the data rather than to create a real federated learning setup with clients on different machines communicating with a server, we simulate the broadcasting step. We reset the weights of the model to the global_weights each time before training with the client data via `client_update` then saving the client models' weights in `local_weights`. Once we have gone through all the clients we aggregrate the weights with weighted averaging and set these as the global model weights.

## Client update

The client update is a normal training setup with respect to the data in the client. We train a model for a small number of epochs and then return the weights and the number of examples in the client (plus some metrics).


```python
def client_update(config, num, model):
    print(num)
    print(pd.DataFrame(pd.read_csv(config.data.train_df_path).query('{}=={}'.format(
        config.data.client_column, num)).label.value_counts()).T)
    dataset = DataGenerator(df_path=config.data.train_df_path, 
                          batch_size=config.data.batch_size, 
                          img_size=config.data.img_size, 
                          n_classes=config.data.n_classes,
                          client_colm=config.data.client_column,
                          num=num)
    history = model.fit_generator(dataset, 
                        epochs=config.train.epochs, 
                        verbose=True,
                        workers=4, 
                        use_multiprocessing=True)
    weights = model.get_weights()
    return (weights,
            len(dataset.filenames),
            history.history['loss'][-1], 
            history.history['acc'][-1])
```

## Weight averaging

Since the weights tend to have defined shapes to update the global weights conveniently we independently average each weight tensor across the sets of weights enabling us to return an aggregated set of weights of the same form as the individual sets of weights.


```python
def average_weights(weights, n_examples):
    weight_lists = map(list, zip(*weights))
    total_examples = np.sum(n_examples)
    return [np.sum(np.stack(w, axis=-1) * n_examples, axis=-1) 
            / total_examples for w in weight_lists]
```
# Results [WIP]

Models are compared based on the number of weight updates needed for the model to reach a certain level of performance. For the standard model each mini-batch leads to an update while for the federated learning models the global model is updated after each communication round. For the randomly sampled shards it takes only 27 rounds to reach the highest accuracy attained by the baseline model of 64.06 % which takes more than 7000 minibatch updates. The model trained on non-representative shard clients did not manage to reach the baseline performance for the number of rounds it was trained. However at lower scores it matches the performance of the baseline after fewer rounds. The baseline reaches above 50% accuracy after more than 1500 updates whereas this model takes a little over 150 rounds. I stopped training as it looked like it was overfitting but the model could potentially be tuned to perform better.

![png]({{ site.baseurl }}/assets/Federated_Farming-Model/acc_plots.png)

Possible overfitting of non-representative shard model.

![png]({{ site.baseurl }}/assets/Federated_Farming-Model/loss_non_iid.png)

![png]({{ site.baseurl }}/assets/Federated_Farming-Model/acc_non_iid.png)

