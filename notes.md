# LLM course from TheHuggingFace

# Transformers Models

### NLP and LLMs

- What are the common NLP tasks ?
    - Classifying whole sentences
    - Classifying each word in a sentence
    - Generating text content
    - Extracting an answer from a texrt
    - Generating a new sentence from an input text
- Is NLP limited to written text ?
    
    No. It also tackles complex challenges in **speech recognition** and **computer vision** (such as generating transcript of an audio sample or a description of an image)
    
- Why has NLP been revolutionized recently ?
    
    Because of LLMs (these models include models like GPT (Generative Pre-Trained Transformers) and Llama
    
- What are the characteristics of LLMs ?
    - **Scale:** They contain millions, billions or even hundreds of billions of parameters
    - **General capabilities**: They can perform multiple tasks without a task-specific training
    - **In-context learning:** They can learn from examples provided in the prompt
    - **Emergent capabilities:** As the model grow in size, they demonstrate capabilities that weren’t explicitly programmed or anticipated
- What are the important limitations of LLMs ?
    - **Hallucinations:** They can generate incorrect informations confidently
    - **Lack of true understanding:** They lack true understanding of the world and operate purely on statistical patterns
    - **Bias:** They may reproduce biais present in their training data or inputs
    - **Context windows:** They have limited context windows (though it is improving)
    - **Computational resources:** They require significant computationnal resources
- Why is NLP challenging ?
    
    

### Transformers, what can they do ?

- What kind of tasks can transformers do ?
    - NLP
    - Computer Vision
    - Processing
- What is the most basic object in the pipeline library ?
    
    the pipeline() function
    
- What does the pipeline() function do when you call it ?
- What are the 3 main steps that are done when you pass a text in a pipeline() ?
    1. The text is preprocessed into a format the model can understand
    2. The preprocessed inputs are passed to the model
    3. The predictions of the model are post-processed, so you can make sense of them
- With what kind of data does the pipeline function works with ?
    - The pipeline() function supports multiple modalities, allowing you to work with text, images, audio and even multimodal tasks
    
    Here is an overview of what’s available
    
    - Text pipelines
        - text-generation: Generate a text from a prompt
        - text-classification: Classify a text into prefined categories
        - summarization: Create a shorter version of a text while preserving key information
        - translation: Translate text from one language to another
        - zero-shot-classification: Classify a text without prior training on specific labels
        - feature-extraction: Extract vector representations of a text
    - Image pipelines
        - image-to-text: Generate text descriptions of images
        - image-classification: Identify objects on a image
        - object detection: Locate and identify objects in images
    - Audio pipelines
        - automatic-speech-recognition: Convert speech to text
        - audio-classification: classify audio into categories
        - text-to-speech: Convert text to spoken audio
    - Multimodal pipelines
        - image-text-to-text: Respond to an image based on a text prompt
- What is the idea behind mask-filling ?
    
    To fill the blanks in a given text
    
- Cite one powerful application of transformers
    
    They are able to **combine and process data from multiple sources**. Especially useful when you need to:
    
    - Search across multiple database and repositories
    - Consolidate information from different formats (text, image, audio)
    - Create a unified view of related information
    - Examples
        
        You could build a system that:
        
        - Search for information in multiple databases in multiple modalities like text and image
        - Combine results from different sources into a single coherent response(e.g. from an audio and a text description)
        - Presents the most relevant information from a database of documents and metadata

### How do Transformers work ?

- When was the Transformers architecture was originally introduced ?
    
    In 2017 and the focus was on translations tasks
    
- How what have been trained the following Transformers: GPT, BERT, T5,…) ?
    
    On language models: they have been trained on large amounts of raw text in a **self-supervised fashion**
    
    - What does self-supervised mean ?
        
        It’s a type of learning in which the objective is automatically computed from the inputs of the model → humans don’t need to label the data
        
    - What does the model do ?
        
        This type of model develops a statistical understanding of the language it has been trained on, but it’s less useful for specific practical tasks
        
        - How to make our model performs well on a specific task ?
            
            The pretrained model then goes through a process called **transfer learning** or **fine-tunning** (during this process, the model is fine-tuned in a supervised way - i.e. using human-annotated labls - on a given task)
            
    - Example a task
        
        Predicting the next word in a sentence the *n* previous words
        
        - Casual language modelling
        - Masked language modelling
        
- How big are Transformers models ?
    
    They are big models (except a few outliers like DistillBERT) → **the general strategy to achieve better performance is by increasing the models’ size as well as the amount on data it has been pretrained on**
    
    - What are the consequences ?
        - Requires a large amount of data
        - Very costly in terms of time and compute resources → translates in big environnemental impact as seen in the following graph
        
- What is pretraining ?
    
    It’s the act of training a model from scratch:
    
    - Weights are randomly initialized
    - The training starts without any prior knowledge
    
    → Usually done on very large amounts of data → it requires a very large corpus and training can takes up to several weeks
    
- What is fine-tunning ?
    
    It’s the training done after the model has been pretrained. To perform fine-tunning, you first acquire a pretrained language model, then perform additionnal training with a dataset specific to your task
    
    - Why not simply train the model for your final use case from the start (**scratch**)?
        - The pretrained model was already trained on a dataset that has some  similarities with the fine-tuning dataset. The fine-tuning process is  thus able to take advantage of knowledge acquired by the initial model  during pretraining (for instance, with NLP problems, the pretrained  model will have some kind of statistical understanding of the language  you are using for your task).
        - Since the pretrained model was already trained on lots of data, the fine-tuning requires way less data to get decent results.
        - For the same reason, the amount of time and resources needed to get good results are much lower
        - Example
            
            For example, one could leverage a  pretrained model trained on the English language and then fine-tune it  on an arXiv corpus, resulting in a science/research-based model. The  fine-tuning will only require a limited amount of data: the knowledge  the pretrained model has acquired is “transferred,” hence the term ***transfer learning***.
            
        
        Fine-tuning a model therefore has lower time, data, financial, and environmental costs. It is also quicker and easier to iterate over different fine-tuning schemes, as the training is less constraining than a full pretraining.
        
        This process will also achieve better results than training from scratch (unless you have lots of data), which is why you should always try to leverage a pretrained model — one as close as possible to the task you have at hand — and fine-tune it.
        
- General Transformer architecture
    - What are the two main block of Transformers ?
        - **Encoder:** The encoder receives an input and builds a representation of it (its feature). This means that the model is optimized to acquire understanding from the input.
        - **Decoder:** The decoder uses the encoder’s representation (features) along with other inputs to generate a target sequence. This means that the model is optimized for generating outputs.
        - Can these parts be used independently ?
            
            Yes, depending on the task
            
            - **Encoder-only models:** Good for tasks that require understanding from the input, such as sentence classification and named entity recognition
            - **Decoder-only models:** Good for generative tasks such as text generation
            - **Encoder-decoder models** or **sequence-to-sequence models**: Good for generative tasks that require an input, such as translation or summarization
    - Attention layers
        
        A key feature of Transformer is that they are built with special layers called ***attention layers***. We will study these layers later in the course but the main thing to know here is that this layer will tell the model **to pay specific attention to certain words in the sentence you passed it** (and more or less ignore the others) when dealing with the representation of each word. 
        
        - Example
            
            You want to translate a text from English to French. 
            
            To put this into context, consider the task of translating text from English to French. Given the input “You like this course”, a translation model will need to also attend to the adjacent word “You” to get the proper translation for the word “like”, because in French the verb “like” is conjugated differently depending on the subject. The rest of the sentence, however, is not useful for the translation of that word. In the same vein, when translating “this” the model will also need to pay attention to the word “course”, because “this” translates differently depending on whether the associated noun is masculine or feminine. Again, the other words in the sentence will not matter for the translation of “course”. With more complex sentences (and more complex grammar rules), the model would need to pay special attention to words that might appear farther away in the sentence to properly translate each word.
            
        
        A word itself has meaning but that meaning is deeply affected by the context (which can be any other word (or words) before or after the word being studied 
        
    - What was the original architecture of Transformers ?
        
        It was originally designed for **translation**. During training:
        
        - The encoder receives inputs (sentences) in a certain language
            - Here, the attention layers c**an use all the words of the sentence** (since translation can be context-dependent)
        - The decoder receives the same sentences in the desired target language
            - The decoder works **sequentially** and can only pay attention to the words in that sentence that has already been translated (so, only the words before the word currently being generated)