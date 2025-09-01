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

    