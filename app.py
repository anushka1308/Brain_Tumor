import dash
from dash.dependencies import Input, Output, State
from dash import dcc
from dash import html
from dash import no_update
import dash_bootstrap_components as dbc
import base64
import io
from io import BytesIO
import re
from keras.models import load_model
from PIL import Image
import numpy as np
from sklearn.metrics import confusion_matrix

# Load model once to avoid reloading in each callback
model = load_model('model/model_final.h5')

# Dummy data for accuracy calculation
y_true = [0, 0, 1, 0, 0, 1]
y_pred = [0, 0, 0, 0, 0, 1]

cm = confusion_matrix(y_true, y_pred)
accuracy = np.sum(np.diag(cm)) / np.sum(cm)

# Define class names and function to interpret prediction
classes = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

def names(number):
    if number == 0:
        return 'a Glioma tumor'
    elif number == 1:
        return 'a Meningioma tumor'
    elif number == 2:
        return 'no tumor'
    elif number == 3:
        return 'a Pituitary tumor'

external_stylesheets = [dbc.themes.BOOTSTRAP]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

app.layout = html.Div([
    dcc.Store(id='upload-store', data={'uploaded': False}),
    
    html.H1(children='BRAIN TUMOR CLASSIFIER', style={'textAlign': 'center'}),
    
    dcc.Markdown('''
        ###### Step 1: Import a single image using the upload button
        ###### Step 2: Wait for prediction and interesting facts to appear
        
        ''',
        style={'font-family': 'Cambria, sans-serif', 'font-size': '30px', 'marginLeft': '20px', 'marginBottom': '20px'}),
    
    dcc.Upload(
        id='upload-image',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '95%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px',
            'backgroundColor': '#343a40',
            'color': 'lightblue'
        },
        multiple=False
    ),
    
    html.Div(id='output-image-upload', style={'position':'absolute', 'left':'200px', 'top':'250px'}),
    html.Div(id='prediction', style={'position':'absolute', 'left':'800px', 'top':'310px', 'font-size':'x-large'}),
    html.Div(id='prediction2', style={'position':'absolute', 'left':'800px', 'top':'365px', 'font-size': 'x-large'}), 
    html.Div(id='facts', style={'position':'absolute', 'left':'800px', 'top':'465px', 'font-size': 'large', 'height': '200px', 'width': '500px'}),
    html.Div(f'Accuracy: {round(accuracy * 100, 2)}%', style={'position': 'absolute', 'top': '50px', 'right': '30px', 'font-size': 'large'})
])

def parse_contents(contents):
    return html.Img(src=contents, style={'height':'450px', 'width':'450px'})

@app.callback(
    [Output('output-image-upload', 'children'), 
     Output('prediction', 'children'), 
     Output('prediction2', 'children'), 
     Output('facts', 'children'),
     Output('upload-store', 'data')],
    [Input('upload-image', 'contents')]
)
def update_output(contents):
    if contents is not None:
        children = parse_contents(contents)
        
        img_data = contents
        img_data = re.sub('data:image/[^;]+;base64,', '', img_data)
        img_data = base64.b64decode(img_data)
        
        stream = io.BytesIO(img_data)
        img_pil = Image.open(stream)
        
        dim = (150, 150)
        img = np.array(img_pil.resize(dim))
        x = img.reshape(1, 150, 150, 3)
        
        answ = model.predict(x)
        classification = np.argmax(answ)
        pred = f"{round(answ[0][classification]*100, 3)}% confidence there is {names(classification)}"
        
        if classification in [0, 1, 3]:
            facts_dict = {
                0: 'Glioma is a type of tumor that occurs in the brain and spinal cord. A glioma can affect your brain function and be life-threatening depending on its location and rate of growth.',
                1: 'A meningioma is a tumor that arises from the meninges, the membranes that surround your brain. Most meningiomas grow very slowly, often over many years without causing symptoms.',
                3: 'Pituitary tumors are abnormal growths that develop in your pituitary gland. Most pituitary tumors are noncancerous (benign) growths that remain in your pituitary gland or surrounding tissues.'
            }
            facts = facts_dict[classification]
            no_tumor_confidence = round(answ[0][2]*100, 3)
            pred2 = f"{no_tumor_confidence}% confidence there is no tumor"
        else:
            facts = None
            pred2 = None
        
        return children, pred, pred2, facts, {'uploaded': True}
    else:
        return no_update, no_update, no_update, no_update, {'uploaded': False}

@app.callback(
    Output('upload-image', 'style'),
    Input('upload-store', 'data')
)
def toggle_upload_box(data):
    if data['uploaded']:
        return {'display': 'none'}
    return {
        'width': '95%', 'height': '60px', 'lineHeight': '60px', 'borderWidth': '1px',
        'borderStyle': 'dashed', 'borderRadius': '5px', 'textAlign': 'center',
        'margin': '10px', 'backgroundColor': '#343a40', 'color': 'lightblue'
    }

if __name__ == '__main__':
    app.run_server(debug=True)
