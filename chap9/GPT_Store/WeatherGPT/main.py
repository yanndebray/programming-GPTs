from flask import Flask, request, Response, send_from_directory
from urllib.request import urlopen
import json, datetime, os

# Api key for openweathermap.org can be provided as an environment variable
app_id = os.getenv('api_key')
app = Flask(__name__)


def validate_api_key(request):
  auth_header = request.headers.get('Authorization')
  if not auth_header:
    return False

  try:
    auth_type, provided_api_key = auth_header.split(None, 1)
    if auth_type.lower() != 'basic':
      return False

    return provided_api_key == app_id
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

  url = f'http://api.openweathermap.org/data/2.5/weather?q={city}&appid={app_id}&units=imperial'
  response = urlopen(url)
  data = response.read().decode('utf-8')

  if data is None:
    return Response(json.dumps(
      {"error": "No current data available for the given city"}),
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


@app.route('/openapi.json')
def serve_openapi_yaml():
  return send_from_directory('.', 'openapi.json', mimetype='application/json')


@app.route('/privacy')
def serve_privacy():
  return send_from_directory('.', 'privacy', mimetype='text')


if __name__ == "__main__":
  app.run(
    host='0.0.0.0',
    port=8080,
    # debug=True,
  )
