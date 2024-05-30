from waitress import serve
import app
serve(app.my_app, host='0.0.0.0', port=8082, threads=4)