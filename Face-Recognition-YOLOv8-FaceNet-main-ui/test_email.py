import smtplib
from email.message import EmailMessage

EMAIL_SENDER = "kommanapavani12@gmail.com"
EMAIL_PASSWORD = "qnhjqdgddzavyucq"
EMAIL_RECEIVER = "rukminivelidi5758@gmail.com"

msg = EmailMessage()
msg["Subject"] = "TEST EMAIL – Face Recognition Project"
msg["From"] = EMAIL_SENDER
msg["To"] = EMAIL_RECEIVER
msg.set_content("If you received this, Gmail SMTP works correctly.")

server = smtplib.SMTP("smtp.gmail.com", 587)
server.starttls()
server.login(EMAIL_SENDER, EMAIL_PASSWORD)
server.send_message(msg)
server.quit()

print("✅ Test email sent successfully")
