import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from django.conf import settings
def send_email(mailID,frame):
    msg = MIMEMultipart('related')
    msg['Subject'] = "Mask violation by you!"
    msg['From'] = settings.EMAIL_HOST_USER
    msg['To'] = mailID
    html = """\
    <html>
    <body>
    <h1 style="background-color: red;text-align: Center;color: white;font-size:40px"> Please wear a mask!</h1>
    <p style="text-align: center;font-size: 30px"> <b>You have been found violating the face mask rules!</b></p>
    <img src="cid:image1" style="display: block;margin-left: auto;margin-right: auto; width:300px ; height:300px;"></img>
    <p>Regards,</p><p>Mask Detector</p>
    </body>
    </html>
    """
    # Record the MIME types of text/html.
    part2 = MIMEText(html, 'html')

    # Attach parts into message container.
    msg.attach(part2)

    # This example assumes the image is in the current directory
    msgImage = MIMEImage(frame)

    # Define the image's ID as referenced above
    msgImage.add_header('Content-ID', '<image1>')
    msg.attach(msgImage)

    # Send the message via local SMTP server.
    server=smtplib.SMTP('smtp.gmail.com',587)
    server.ehlo()
    server.starttls()
    server.ehlo()
    server.login(settings.EMAIL_HOST_USER,settings.EMAIL_HOST_PASSWORD)
    server.sendmail(settings.EMAIL_HOST_USER, mailID, msg.as_string())
    server.quit()
    
