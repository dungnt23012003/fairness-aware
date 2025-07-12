# Fairness-aware-synthetic-financial-data-using-GAN

## Steps to run Code
- Clone the repository or download from the release on the right side
```
https://github.com/dungnt23012003/fairness-aware.git
```

- Go to the cloned folder.
```
cd fairness-aware
```

- Create a virtual envirnoment (Recommended, If you dont want to disturb python packages)
```
### For MacOS/Linux Users
source venv/bin/activate

### For Window Users
cd venv
cd Scripts
activate
cd ..
cd ..
```

- Upgrade pip with mentioned command below.
```
pip install --upgrade pip
```
- Install requirements with mentioned command below.
```
pip install -r requirements.txt
```

- Clean your data before use, to make sure:
    1. Any column name cannot fully include anther one (E.g. 'education', 'education_number')
    2. Use epochs number of 50, 100, 200, do not use other numbers, default is 50
    3. No nan value in the data, check with "dataframe.isnull().values.any()"

- Move your data to the 'data' folder

- Run the code below to train and generate.

```
python3 run.py --source 'olympics.csv' \
               --output 'olympics_generation.csv' \
               --amount 1.0 \
               --continuous_columns "Age" "Height" "Weight" \
               --categorical_columns 'Sex' 'Year' 'Season' 'City' 'Sport' 'Medal' 'AOS' 'AOE' 
```
- Please find your generated data in "generations" folder

- An evaluation and visualization example is in the file "evaluation_visualization.ipynb"

- Visualization examples are in the folder "visualizations"

- Grid search is recommended if your performance is not satisfying. Parameters that can be turned: batch-size, epochs, learning rate, leakyRely negative slope of generator, leakyRely negative slope of discriminator

