# Recurrent Neural Networks

## What Makes a Neural Network Recurrent?

### Introduction
In this module, you’ll learn **Recurrent Neural Networks** or **RNNs**. RNNs are specially designed to work with **sequential data**, i.e. data where there is a natural notion of a 'sequence' such as text (sequences of words, sentences etc.), videos (sequences of images), speech etc. RNNs have been able to produce state-of-the-art results in fields such as natural language processing, computer vision, and time series analysis.

One particular domain RNNs have revolutionised is **natural language processing**. RNNs have given, and continue to give, state-of-the-art results in areas such as machine translation, sentiment analysis, question answering systems, speech recognition, text summarization, text generation, conversational agents, handwriting analysis and numerous other areas. In computer vision, RNNs are being used in tandem with CNNs in applications such as image and video processing.

Many RNN-based applications have already penetrated consumer products. Take, for example, the **auto-reply** feature which you see in many chat applications, as shown below:

![title](img/auto-reply.JPG)

You may have noticed auto-generated subtitles on YouTube (and that it has surprisingly good accuracy). This is an example of **automatic speech recognition (ASR)** which is built using RNNs.

Similarly, when you talk to a support team of a food delivery app, or any other support team for that matter, you get an auto-reply in the initial stages of your interaction where the support team asks about details such as order date, problem description and other basic things. Many of these conversational systems, informally called '**chatbots**', are trained using RNNs.

RNNs are also being used in applications other than NLP. Recently, OpenAI, a non-profit artificial intelligence research company came really close to defeating the world champions of Dota 2, a popular and complex battle arena game. The game was played between a team of five bots (from OpenAI) and a team of five players (world champions). The bots were trained using reinforcement learning and recurrent neural networks.

There are various companies who are generating music using RNNs. [Jukedeck](https://www.jukedeck.com/about) is one such company.

There are many other problems which are yet to be solved, and RNNs look like a promising candidate to solve them. You could be the one to make an impact in those areas. With that in mind, let’s start the module.

### What are Sequences?
Just like CNNs were specially designed to process images, **Recurrent Neural Networks (RNNs)** are specially designed to process **sequential data**. 

In **sequential data**, entities occur in a particular order. If you break the order, you don’t have a meaningful sequence anymore. For example, you could have a sequence of words which makes up a document. If you jumble the words, you will end up having a nonsensical document. Similarly, you could have a sequence of images which makes up a video. If you shuffle the frames, you’ll end up having a different video. Likewise, you could have a piece of music which comprises of a sequence of notes. If you change the notes, you’ll mess up the melody.

Recurrent neural networks are variants of the vanilla neural networks which are tailored to learn sequential patterns. 

![title](img/sequential_data.JPG)

You saw some examples of sequence problems. Let’s now see an interesting unconventional example of a problem involving sequences which can be solved using RNNs.

![title](img/example_sequential_data.JPG)

![title](img/rnn.JPG)

Although sorting is a problem that involves an algorithm, but the fact that RNNs can learn an algorithm speaks volume about their capacity to solve hard learning problems.

### What Makes the Network Recurrent
You now know what sequences are. In this section, you’ll learn how a normal feedforward network is modified to work with sequences.

Let's quickly recall the feedforward equations of a normal neural network:

![title](img/rnn1.JPG)

Let's now understand what makes this specialised neural network 'recurrent'.

![title](img/rnn2.png)

![title](img/rnn4.JPG)

These notations and ideas will be more clear going forward. Let's now look at the feedforward equations of an RNN in the following segment.

![title](img/feed_forward_rnn.JPG)

### Architecture of an RNN
Let's now look at how the architecture of an RNN visually and compare it to a normal feedforward network.

![title](img/rnn_architecture.png)

The following figure shows the RNN architecture along with the feedforward equations:

![title](img/rnn5.JPG)

![title](img/rnn6.JPG)

For example, let's say that layer-2 and layer-3 have 10 and 20 neurons respectively. Each of the red copies of the second layer will have 10 neurons, and likewise for layer-3.  

![title](img/rnn7.JPG)


### Feeding Sequences to RNNs
In a previous segment, we had discussed sequences briefly. Let’s now take a look at how various types of sequences are fed to RNNs.

Now that you understand how an RNN consumes one sequence, let’s see how do you train a batch of such sequences.

You learnt how to feed data to an RNN. In the case of an RNN, each data point is a sequence.The individual sequences are assumed to be **independently and identically distributed (I.I.D.)**, though the entities within a sequence may have a dependence on each other.

![title](img/rnn8.JPG)

You also saw how the data can be fed in **batches** just like any normal neural net - a batch here comprises of multiple data points (sequences).

### Comprehension: RNN Architecture
In a previous section, you studied the feedforward equations of RNNs. Let's now analyse the architecture in a bit more detail - we will compute the dimensions of the weight and bias matrices, the outputs of layers etc. We will also look at a concise form of writing the feedforward equations.

As you already know, the architecture of an RNN and its feedforward equations are as follows:

![title](img/rnn5.JPG)

![title](img/rnn9.JPG)

![title](img/rnn10.JPG)

![title](img/rnn11.JPG)

### RNNs: Simplified Notations
You may commonly come across a concise, simplified notation scheme for RNNs. Let's discuss that as well. The RNN feedforward equations are:

![title](img/rnn12.JPG)

![title](img/rnn13.JPG)

This form is not only more concise but also more computationally efficient. Rather than doing two matrix multiplications and adding them, the network can do one large matrix multiplication. 

Now consider the same example with the modified notations in mind. You have a neural network with three neurons in the input layer (layer-0), 7 neurons in the hidden layer (layer-1) and one neuron in the output softmax layer (layer-2). Consider a batch size of 64. The sequence size is 10.

![title](img/q1.JPG)

![title](img/ans1.JPG)

![title](img/q2.JPG)

![title](img/ans2.JPG)

### Types of RNNs - I
In the previous few segments, you studied the architecture of RNNs. You saw that there is an input sequence fed to the input layer, and an output sequence coming out from the output layer. The interesting part is that you can change the sizes and types of the input-output layers for different types of tasks. Let’s discuss some commonly used RNN architectures:

* **Many-to-one RNN**
In this architecture, the input is a sequence while the output is a single element. We have already discussed an example of this type - classifying a sentence as grammatically correct/incorrect. The figure below shows the **many-to-one** architecture:

![title](img/many-to-one.JPG)

Note that each element of the input sequence **x<sub>i</sub>** is a numeric vector. For words in a sentence, you can use a one-hot encoded representation, use word embeddings etc. You’ll learn these techniques in the next session. Also, note that the output is produced after the last timestep T (after the RNN has seen all the inputs).

 

