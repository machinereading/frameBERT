
import os
import sys
sys.path.append('../')

from flask import Flask
from flask_restful import Api, Resource, reqparse

from frameBERT import frame_parser

app = Flask(__name__)
api = Api(app)

# import jpype
# jpype.attachThreadToJVM()

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True)
parser.add_argument('--language', required=False, default='ko')
parser.add_argument('--port', required=False, default=1106)
args = parser.parse_args()

# In[1]:

f_parser = frame_parser.FrameParser(model_path=args.model, masking=True, language=args.language)

class WebService(Resource):
    def __init__(self):
#         self.parser = frame_parser.FrameParser(model_path=args.model, masking=True, language=args.language)
    def post(self):
        try:
            req_parser = reqparse.RequestParser()
            req_parser.add_argument('text', type=str)
            req_parser.add_argument('sent_id', type=str)
            req_parser.add_argument('result_format', type=str)
            args = req_parser.parse_args()
            print(args)
            if not args['sent_id']:
                sent_id = False
            else:
                sent_id = args['sent_id']
            if not args['result_format']:
                result_format = 'graph'
            else:
                result_format = args['result_format']
                
            result = f_parser.parser(args['text'], sent_id=sent_id, result_format=result_format)

            return result, 200
        except KeyboardInterrupt:
            raise
        except Exception as e:
            return {'error':str(e)}

api.add_resource(WebService, '/frameBERT')
app.run(debug=True, host='0.0.0.0', port=args.port)

