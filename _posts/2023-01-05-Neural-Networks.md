---
layout: page
title: Creating a Niche Classifier with Neural Networks
permalink: /neural-networks/
---

### Introduction
The wedding industry consists of many independent designers, artists, and planners who are responsible for every aspect of their businesses. An important aspect of their marketing efforts takes place on social media platforms like Instagram. Over time, they develop large libraries of pictures that are the core of marketing efforts. These pictures come unlabeled and a great amount and it takes a great amount of time to search through them to select images for posts. Selecting photos for marketing takes valuable time away from spending time working with their clients. Computer vision techniques using neural networks offer a solution to this problem: using established image classification neural network models can I employ transfer learning to fine tune a model to create a niche classifier to classify wedding images on a small dataset?

[Find the repo on github](https://github.com/russ-kibat/neural-networks)

### Data Background
Image files are difficult to search because elements of the image are represented by numerical pixel data. Computer vision has come of age with the development of machine learning and neural networks. Great effort has been devoted to building classification neural networks. Many of the most powerful neural networks are general classifiers. This makes them great at high level distinctions when classifying, but they need to be fine tuned to make finer distinctions between similar classes. 

While image file formats have the potential to store metadata, useful information regarding the content is missing unless it is added manually. Before computer vision was developed this kind of classification could only be done though a manual time intensive process of sorting and labeling images. Now, with better models and access to more computing power it is possible for these models to be fine tuned for niche applications.
The data used for this project consists of a collection of images downloaded from Instagram. I was able to collect images from wedding-related hashtags and build a dataset for this model. The classification  targets for the model are: bride, bouquet, newlyweds, reception room, and wedding party. Instagram users often place many general or unrelated tags on their posts. Starting with 10,000 image downloads, multiple stages of sorting and refining yielded a dataset of 2,500 images. 

![Target-Class-Examples](/images/Example-Classes.png)

I sorted into groups by class. These classes then needed to be further refined. Many images may have multiple classes represented in them. In order to train effectively I removed images that did not exemplify the class. Two classes, bouquet and bride, often appear together so it is important to have sets that only include one or the other item. I also removed images with poor lighting.

![Data-Collection-Pyramid](/images/Data-Collection.png)

### Modeling and Evaluation
With a small dataset, I wanted to use top models that used a relatively small amount of trainable parameters. I started with the base models of ResNet50V2 and MobileNetV2 running each of them with only a customized final classification layer added. The initial results for both models were impressive. To improve the models I ran several iterations adding a few layers to the networks and adjusting hyperparameters. 
I tracked two while evaluating models: training loss, how well the model fits the data, and F1 score, a measure of prediction accuracy. Gains were achieved when image augmentation layers were added. These layers flipped and rotated training images to allow the model to identify features in a variety of positions in the image to enhance the models ability to identify them in more locations on unseen data. Lowering the model learning rate also contributed to improvements. The final model is a version of the ResNet50V2 model with these additions.

### Conclusion
My results challenged my assumptions on both the potential shortfalls of state of the art image classification neural networks and the quality and quantity of data it takes to improve upon them. A larger dataset could lead to a more robust and more useful model. That would also allow for more opportunities to fine tune the base models.

The dataset can expand in two directions. First adding more images that exemplify the classes create cleaner training data. I would also like to add more classes to expand the reach of the model. The images I collected were also from a very small window in time and from a very small potential source. Reaching out to well established photographers would be the next source I would like to investigate. Not only would a collection of their career portfolio span a longer timeline, but they will also include a variety of compositions that make training more robust. It’s also important to acknowledge that my approach does not take into account the variety of wedding traditions and customs. The first approach to handling this issue would be to limit the scope of the model to a particular segment of weddings that my target users work in. With more data and training that could be built upon to include a broader range of users. 

While this model has room for improvement, implementing it into an application that can classify batches of images does have value. Automating this classification process will save valuable time in selecting appropriate images to post. Selecting images for marketing posts ultimately takes the expertise and attention of the user. Removing the burden of labeling or starting with an unlabeled library for every search will greatly reduce the time required to select these images. Leveraging current neural network models and processing power does offer intriguing possibilities for niche projects.   

[Find the repo on github](https://github.com/russ-kibat/neural-networks)