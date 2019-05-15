<strong>Classifying Pokemon using a Deep Learning Neural Network</strong><br></br>
I've been playing the Pokemon video games for over a decade and they've brought me literally hundreds of hours of entertainment.  However, I sometimes still struggle to correctly identify a Pokemon's type.  This can be critical to successfully completing the game as type will dictate which Pokemon are most effective against others and which will be weak to certain attacks.  For example, Ice-type Pokemon are weak to attacks from Fire-type Pokemon.  For my project, I wanted to see if a neural network could be trained to effectively identify a Pokemon's type based on unsupervised learning of discreet features of Pokemon images.  While there are a number of features that could be used to potentially identify a Pokemon's type, I chose images as some Pokemon are rightly considered iconic characters.  In addition, the designers of Pokemon have frequently made considerable efforts to incorporate features related to the type of a Pokemon into the design of individual Pokemon.  I wanted to see if these signifiers could be interpreted by a neural network and, if so, how accurately?
<br></br>
<strong>Data Collection</strong><br></br>
I started with a dataset retrieved from a Kaggle kernel (https://www.kaggle.com/abcsds/pokemon) that listed all the currently-released Pokemon. An addition kernel (https://www.kaggle.com/kvpratama/pokemon-images-dataset) provided single images of each.  The resultant dataset contained around 800 images and Type labels for each image.  As I knew that computer vision problems (and most data science problems in general) benefit from more examples of each class,  I consulted a tutorial (add tutorial link here) on web scraping for images and collected around 1000 additional image examples of each type of Pokemon. Additional literature review led me to understand that this was still woefully insufficient for an image classification problem since, in most image classification problems, at least 1000 images per class are recommended for a small dataset.  With 18 unique Pokemon types, I'd need to collect significantly more data. I revised the web scraper to collect approximately 30 images for each Pokemon species which resulted in about 25000 images in the final dataset.  The web scraper uses the Azure Cognitive Services API so you'll need an API key to use the scraper script.
<br></br>
<strong>Analysis</strong><br></br>
Familiarity with the system from prior trainings and exposure led to me using the keras front-end for Tensorflow to build the image classification model.  Also based on prior familiarity with working with data in pandas dataframes, I used an ImageDataGenerator and the flow_from_dataframe method to feed data to the model. I started with a homebrew model composed of a few convolutional layers using a relu activation function to avoid negative weighting of features during training.  I experimented with various neuron densities and depths of layers but even a well-trained model using tiny batches of images struggled to achieve an accuracy of even 1%.  In need of a more robust approach, I investigated using transfer learning to pre-train the model for image classification.  Luckily, keras provides an applications repository that contains various pre-trained models specifically for image-recognition problems.  Combining that knowledge with a tutorial I followed to integrate the pre-trained model with my dataset and its collection of pre-defined classes, I was able to experiment with different pre-trained models and various hyperparameters to create a much more accurate model.  The greatest degree of accuracy came from using the ResNet50 model with imagenet weights as a frozen layer and piping the resulting weights into a handful of fully connected layers that that ended in a Dense layer of 18 neurons (1 per class) giving the predicted classifications of each image.  I had the best results when using tiny batches of 8 images at a time and running each batch through 1250 steps per epoch to extract a large amount of information from each image before moving to the next iteration of training.  This achieved about 67% accuracy in classifying an image.  However, the granularity of training at this rate caused the training process to be extremely time-consuming.  The best results were achieved after a training run that lasted in excess of 16 hours.  Further tuning should be done to achieve a good balance between performance and time required to iterate on the model.
<br></br>
A link to a slide presentation on the project can be found here: https://docs.google.com/presentation/d/12MKYYJOdIZYVdkqCr31FH9Qb9HuUuLCBpwygXQpo78U/edit?usp=sharing
