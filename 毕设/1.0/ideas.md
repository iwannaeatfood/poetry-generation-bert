April 21
Zhiyao Liang
Yanyan Zhu
Ruoti Zhang

Q and A: 
- Question have different kind: 
    - Given some text, answer qustions. (reading comprehension)
    - Question without background text: 
        - Example:  
            - Is an elephant larger than a cat ? 
            wordnet? 
            common knowledge database? 
            - How can we put an elephant into a refrigrator?
- Questions have different levels: 
    - Reading comprehension:
        1) What are the locations? Person Names? Time points? Events? 
        2) Relationships between items. 
            Who did what? when? how? why? 
        3) Deeper logical relationships. 

- Guide the computation process. 
  - step 1)  build the basic information base: 
     perons, locations, time points, actions events 
  - step 2)  build the relationships 
       sombody did somthing at ... 
       <person> ----- <action> ------- <target>
  - step 3) parse the question. 
  - step 4)  final touch 

  What is obama's wife's favouriate flower. 



Ideas of making progress: 
- Anything that make difference is good. 

How to measure the quality? 

--------------------

Text generation

Different kinds: 
- Poems 
    - traditional chinese 
        - 5 words, 7 words, Song style 
    - mordern chinese 
- Given some words, make sentence: 
    - Example: Given "macau", "flower", "science",  make sentence. 
- Given some suggestion, make some text 
    - news broadcast
    - weather news. 
- Generate some questions based on some text or information. 
- Generate bad poems. 
- More requirements of text generation: 
    - number of words
    - emtional style 
    - rhyme 
    - logical correctness 
    - style, deeper effects
- Which task is easier? 
    - generate the plan of the sentences vs real sentences. 
    - generate one sentence vs a poem 
- Guide the computation process. 
  - step 1)  write the plan or ideas of the four sentences. 
  - step 2)  schedule some rythme policy 
  - step 3)  tranlsate each sentence plan into some real sentence 
  - step 4)  final touch 

How to measure the quality? 
 - Generate questions with different difficulty levels. 

----------------------------
Plan
1)  build the program that can talk with the AI model 
     - The output of the AI can be stored in some datastructure of the program. 
     - The program can repeatedly talk with the AI. 
     - Some question/request to AI, must be generated/built by your program. 