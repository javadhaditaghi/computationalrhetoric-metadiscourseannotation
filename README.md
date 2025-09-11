How to run this code


1. Run the annotators codes one by one 
   Run deepseek
      ```src/annotation/llm/annotators/deepseek_annotator.py test_restricted.csv  --context-before=30 --context-after=30```
   
   Run claude
    ```src/annotation/llm/annotators/claude_annotator.py test_restricted.csv  --context-before=30 --context-after=30```
    
    Run gemini
    ```src/annotation/llm/annotators/gemini_annotator.py test_restricted.csv  --context-before=30 --context-after=30```

2. Run the setup
   
    ```python setup.py  ```
3. Run the optimizer annotator

   ```python run_optimizer.py --context-before 30 --context-after 30 ```

4. Flatten the optimized json file

   ```python src\postprocessing\flatter-annotation.py ```

5. It's time to jump into the analysis folder and use jupyter for comparing LLM annotated dataset and Human annotated dataset.

    * _Calculating IAA, precision, recall, accuracy, F1 score, cofusion matrix, chi-square test, p-value_

6. Run the prompt improvement file (The number of rows can be specified)

    ```python src/postprocessing/promptimprovement_1.py --rows 10 ```
    