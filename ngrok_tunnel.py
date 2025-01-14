from pyngrok import ngrok
import getpass

# Get ngrok auth token from ngrok website (https://dashboard.ngrok.com)
print("Enter your ngrok auth token:")
auth_token = getpass.getpass()  # Hide password input
ngrok.set_auth_token(auth_token)

# Open a tunnel on port 5000 (for example)
public_url = ngrok.connect(5000).public_url

# Output the public URL
print(f' * ngrok tunnel "{public_url}"')