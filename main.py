import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

class Main:
		def __init__(self):
				pass
		


		def fill_out_missing_values(self, df, column):
				
				#Måler occurence i procent for alle de kategoriske variable
				percentage_distribution = df[column].value_counts(normalize=True).to_dict()

				filled_values = []

				for index, row in df.iterrows():
					if pd.isna(row[column]):
							# Generate a random value based on the percentage distribution
							random_value = np.random.choice(list(percentage_distribution.keys()), p=list(percentage_distribution.values()))
							filled_values.append(random_value)
					else:
							filled_values.append(row[column])

				df[column] = filled_values
				print(df[column].head(50))
				return df  # Re

		def preprocess_data(self, df):
			
				# Et imputer object der udfylder manglende værdier baseret på mean af de eksisterende værdier i eks age kolonnen
				imputer = SimpleImputer(strategy='mean')
				df['Age'] = imputer.fit_transform(df[['Age']])
				df['Fare'] = imputer.fit_transform(df[['Fare']])

				encoder = LabelEncoder()
				df['Sex'] = encoder.fit_transform(df['Sex'])
			
				df = self.fill_out_missing_values(df, 'Embarked')
				df['Embarked'] = encoder.fit_transform(df['Embarked'])

				# Selecting relevant features
				features = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
				return features

		def train_and_evaluate(self):
				df = pd.read_csv('train.csv')
				
				#se de første 3 rækker i træningsdataen
				print(df.head(3))
				
				#Pivot

				#Lave et pivot table for at se overlevelsesraten for mænd og kvinder i hver klasse
				pivot_table = df.pivot_table(values='Survived', index='Sex', columns='Pclass', aggfunc='mean')
				print(pivot_table)
				
				#ML
				X= self.preprocess_data(df)
				y = df['Survived']

				# X_train er alle de uafhængige variable, og y_train er den afhængige variabel'Survived', som vi vil forudsige
				# train_test_split træner på 80% af dataen og efterlader 20 % af dataen til at teste på efter, så test data er data der ikke er blevet trænet på. random_state på 42 sikrer at den deler dataen op i de forskellige grupper på samme måde hver gang, for at opnå reliabilitet på tværs af tests
				X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

				# Model training object using decision tree
				model = RandomForestClassifier()#
    		#trains data on model data
				model.fit(X_train, y_train)

				# Predicting and evaluating
				predictions = model.predict(X_test)
				accuracy = accuracy_score(y_test, predictions)
				print(f'Accuracy: {accuracy}')
	
				#Load test data
				test_df = pd.read_csv('test.csv')
				predictions = model.predict(self.preprocess_data(test_df))

				result_df = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Survived': predictions})
				print(result_df.head(10))
				result_df.to_csv('results.csv', index=False)

    
if __name__ == '__main__':
		app = Main()
		app.train_and_evaluate()