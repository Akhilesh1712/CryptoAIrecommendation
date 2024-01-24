from flask import Flask, render_template, request, jsonify
import requests
import pandas as pd
from surprise import Dataset
from surprise import Reader
from surprise import SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
app = Flask(__name__)


# Split the data into training and testing sets
trainset, testset = train_test_split(data, test_size=0.2)

# Train the collaborative filtering model
model = SVD()
model.fit(trainset)

# Set up a list of cryptocurrencies to analyze
cryptos = ['bitcoin', 'ethereum', 'cardano', 'binancecoin', 'dogecoin', 'solana', 'polkadot', 'chainlink', 'ripple', 'litecoin']

# Create an empty DataFrame to store the cryptocurrency data
columns = ['Name', 'Symbol', 'Price (USD)', 'Market Cap (USD)', '24h % Change', '7d % Change', '30d % Change']
df = pd.DataFrame(columns=columns)

# Loop over the list of cryptocurrencies and retrieve data from the CoinGecko API
for crypto in cryptos:
    # Set up the API endpoint URL for the current cryptocurrency
    url = f"https://api.coingecko.com/api/v3/coins/{crypto}"
    
    # Use the requests library to send a GET request to the API endpoint
    response = requests.get(url)
    
    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse the response JSON into a Python dictionary
        data = response.json()
        
        # Extract relevant data for each cryptocurrency
        name = data['name']
        symbol = data['symbol'].upper()
        price = data['market_data']['current_price']['usd']
        market_cap = data['market_data']['market_cap']['usd']
        day_change = data['market_data']['price_change_percentage_24h']
        week_change = data['market_data']['price_change_percentage_7d']
        month_change = data['market_data']['price_change_percentage_30d']
        
        # Append the data to the DataFrame
        df = df.append({'Name': name, 'Symbol': symbol, 'Price (USD)': price, 'Market Cap (USD)': market_cap,
                        '24h % Change': day_change, '7d % Change': week_change, '30d % Change': month_change},
                       ignore_index=True)
    else:
        print(f"Error fetching data for {crypto}. Status code: {response.status_code}")

# Print the DataFrame
print(df)

# Optionally, you can save the DataFrame to a CSV file
df.to_csv('cryptocurrency_data.csv', index=False)
# Assuming your DataFrame is named df
reader = Reader(rating_scale=(0, 1))
data = Dataset.load_from_df(df[['Name', 'Symbol', '30d % Change']], reader)
# Split the data into training and testing sets
trainset, testset = train_test_split(data, test_size=0.2)

# Train the collaborative filtering model
model = SVD()
model.fit(trainset)
# Example: predict the rating for user 'user_id' and item 'item_id'
user_id = 1
item_id = 'bitcoin'
prediction = model.predict(user_id, item_id)

print(f"Predicted rating for {item_id} by user {user_id}: {prediction.est}")
# Make predictions on the test set
predictions = model.test(testset)

# Evaluate the RMSE (Root Mean Squared Error) on the test set
rmse = accuracy.rmse(predictions)
print(f"RMSE on the test set: {rmse}")
@app.route('/recommendations', methods=['POST'])
def get_recommendations():
    # Receive data from the front end
    user_id = int(request.form['user_id'])
    selected_crypto = request.form['crypto']

    # Make a prediction for the selected cryptocurrency
    prediction = model.predict(user_id, selected_crypto)
    predicted_rating = prediction.est

    # Return the result as JSON
    return jsonify({'crypto': selected_crypto, 'predicted_rating': predicted_rating})

if __name__ == '__main__':
    app.run(debug=True)