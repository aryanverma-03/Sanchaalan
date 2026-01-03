# smtp.py
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from email.utils import formataddr
from typing import List, Optional

# Configure your Gmail account
SENDER_EMAIL = "mtyagi2002@gmail.com"
APP_PASSWORD = "jdxd szpf yhpy itim" 

def send_email(subject: str, body: str, recipients: List[str], attachments: Optional[List[str]] = None):
    """
    Send an email with optional attachments.
    """
    msg = MIMEMultipart()
    msg["From"] = formataddr(("Team Sanchaalan", SENDER_EMAIL))
    msg["To"] = ", ".join(recipients)
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    # Attach files
    for file_path in attachments or []:
        with open(file_path, "rb") as f:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", f'attachment; filename="{file_path}"')
            msg.attach(part)

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(SENDER_EMAIL, APP_PASSWORD)
        server.sendmail(SENDER_EMAIL, recipients, msg.as_string())
        print(f"📧 Email sent successfully to {recipients}")
    except Exception as e:
        print("❌ Email error:", e)
    finally:
        server.quit()
