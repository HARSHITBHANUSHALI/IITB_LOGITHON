from flask import Flask, request, jsonify
from twilio.rest import Client

app = Flask(__name__)

account_sid = 'AC338d14a32c3fd7d04aa789471a6617e5'
auth_token = 'c2a0f002936b4a8a62b219f8865ae247'
client = Client(account_sid, auth_token)

@app.route('/send_message', methods=['POST'])
def send_message():
    data = request.json
    from_whatsapp_number = 'whatsapp:+14155238886'
    to_whatsapp_number = data.get('to')
    content_sid = 'HXb5b62575e6e4ff6129ad7c8efe1f983e'
    content_variables = data.get('content_variables')
    message_body = data.get('message')

    message = client.messages.create(
        from_=from_whatsapp_number,
        body=message_body,
        to=to_whatsapp_number
    )

    return jsonify({'message_sid': message.sid})

if __name__ == '__main__':
      app.run(host='0.0.0.0', port=5009, debug=True)