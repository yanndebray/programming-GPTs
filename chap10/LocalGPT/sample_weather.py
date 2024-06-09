from urllib.request import urlopen
import json, datetime
from flask import Flask, request, send_from_directory

city = 'London,uk'
app_id = 'b1b15e88fa797225412429c1c50c122a1'
# Get sample weather data from openweathermap.org
url = f'http://samples.openweathermap.org/data/2.5/weather?q={city}&appid={app_id}'

app = Flask(__name__)


@app.route('/')
def index():
  return "Hello, Weather!"


@app.route('/weather', methods=['GET'])
def get_weather_data():
  q = request.args.get('q')
  response = urlopen(url)
  data = response.read().decode('utf-8')
  json_data = json.loads(data)
  # select data of interest from dictionary
  weather_info = json_data['main']
  weather_info.update(json_data['wind'])
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
