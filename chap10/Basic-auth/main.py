# https://mer.vin/2023/11/get-stock-price-with-basic-auth/
from flask import Flask, request, Response
from urllib.request import urlopen
import json, datetime, os

city = 'London,uk'
app_id = 'b1b15e88fa797225412429c1c50c122a1'
app = Flask(__name__)

API_KEY = "My_API_Key"  # Replace with your actual API key

def validate_api_key(request):
    auth_header = request.headers.get('Authorization')
    if not auth_header:
        return False

    try:
        auth_type, provided_api_key = auth_header.split(None, 1)
        if auth_type.lower() != 'basic':
            return False

        return provided_api_key == API_KEY
    except Exception:
        return False

@app.route('/')
def index():
  return "Hello, Weather!"


@app.route('/weather', methods=['GET'])
def get_weather_data():
    if not validate_api_key(request):
            return Response(json.dumps({"error": "Invalid API key"}),
                            status=401,
                            mimetype='application/json')
    city = request.args.get('city')
    if not city:
        return Response(json.dumps({"error": "City parameter is required"}),
                        status=400,
                        mimetype='application/json')

    url = f'http://samples.openweathermap.org/data/2.5/weather?q={city}&appid={app_id}'
    response = urlopen(url)
    data = response.read().decode('utf-8')

    if data is None:
        return Response(json.dumps({"error": "No current data available for the given city"}),
                        status=404,
                        mimetype='application/json')
    
    json_data = json.loads(data)
    # select data of interest from dictionary
    weather_info = json_data['main']
    # weather_info.update(json_data['wind'])
    weather_info.update(json_data['coord'])
    weather_info['city'] = json_data['name']
    # add current date and time
    weather_info['current_time'] = str(datetime.datetime.now())
    return weather_info


if __name__ == "__main__":
    app.run(
        host='0.0.0.0', 
        port=8080,
        debug=True,
        )
