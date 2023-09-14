# using SendGrid's Python Library
# https://github.com/sendgrid/sendgrid-python
import os
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

message = Mail(
    from_email='contato@wearebren.com',
    # to_emails='bernardo.bicalho@wearebren.com',
    to_emails='bbicalho@gmail.com',
    subject='Sending with Twilio SendGrid is Fun',
    html_content='<strong>and easy to do anywhere, even with Python</strong>')
try:
    sg = SendGridAPIClient('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')
    response = sg.send(message)
    print(response.status_code)
    print(response.body)
    print(response.headers)
except Exception as e:
    print(e)