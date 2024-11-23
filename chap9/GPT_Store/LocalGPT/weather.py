from urllib.request import urlopen
import json, datetime, os
from flask import Flask, request, send_from_directory

# Api key for openweathermap.org can be provided as an environment variable
os.environ['api_key'] = 'My_API_Key'

# city = 'Boston,US'
app_id = os.getenv('api_key')
app = Flask(__name__)


@app.route('/')
def index():
  return "Hello, Weather!"


@app.route('/weather', methods=['GET'])
def get_weather_data():
  city = request.args.get('city', 'Boston,US') # default city is Boston, US
  url = f'http://api.openweathermap.org/data/2.5/weather?q={city}&appid={app_id}&units=imperial'
  response = urlopen(url)
  data = response.read().decode('utf-8')
  json_data = json.loads(data)
  # select data of interest from dictionary
  weather_info = json_data['main']
  # weather_info.update(json_data['wind'])
  weather_info.update(json_data['coord'])
  weather_info['city'] = json_data['name']
  # add current date and time
  weather_info['current_time'] = str(datetime.datetime.now())
  return weather_info


@app.route('/.well-known/ai-plugin.json')
def serve_ai_plugin():
  return send_from_directory('.',
                             'ai-plugin.json',
                             mimetype='application/json')


@app.route('/openapi.yaml')
def serve_openapi_yaml():
  return send_from_directory('.', 'openapi.yaml', mimetype='text/yaml')


@app.route('/privacy')
def serve_privacy():
  return send_from_directory('.', 'privacy', mimetype='text')


if __name__ == '__main__':
  app.run(
    host='0.0.0.0',
    port=8080,
    debug=True,
  )
