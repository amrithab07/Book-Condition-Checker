import os
from app import app

if __name__ == '__main__':
    # Public binding for container
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', '5000'))

    # Ensure debug is off in production
    debug = os.getenv('DEBUG', 'false').lower() in ('1','true','yes')

    app.run(host=host, port=port, debug=debug)

