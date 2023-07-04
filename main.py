import transformers
from transformers import pipeline


#simple  responses

TextClasification = pipeline(task="sentiment-analysis")
response1 = TextClasification("something in here") # will give a lebel to the prompt
print(response1)

TextGeneration = pipeline(task = "text-generation")
response2 = TextGeneration("something in here") #will give a response to given prompt
print(response2)

#you can do the same for

# Image classification	assign a label to an image	Computer vision	pipeline(task=“image-classification”)

# Image segmentation	assign a label to each individual pixel of an image (supports semantic, panoptic, and instance segmentation)	Computer vision	pipeline(task=“image-segmentation”)

# Object detection	predict the bounding boxes and classes of objects in an image	Computer vision	pipeline(task=“object-detection”)

# Audio classification	assign a label to some audio data	Audio	pipeline(task=“audio-classification”)

# Automatic speech recognition	transcribe speech into text	Audio	pipeline(task=“automatic-speech-recognition”)

# Visual question answering	answer a question about the image, given an image and a question	Multimodal	pipeline(task=“vqa”)

# Document question answering	answer a question about a document, given an image and a question	Multimodal	pipeline(task=“document-question-answering”)

#Image captioning	generate a caption for a given image	Multimodal	pipeline(task=“image-to-text”)



## More documentation for these and install directions can be found at https://huggingface.co/docs/transformers/quicktour
