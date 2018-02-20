import os
from OpenSSL import SSL
from bottle import ServerAdapter, response, request

try:
    from cheroot.wsgi import Server as WSGIServer
    from cheroot.ssl.pyopenssl import pyOpenSSLAdapter
except ImportError:
    from cherrypy.wsgiserver import CherryPyWSGIServer as WSGIServer
    from cherrypy.wsgiserver.ssl_pyopenssl import pyOpenSSLAdapter


# TODO(kelvin): set these absolute file paths (str)
BASEPATH = os.environ['CERT_BASEDIR']
SSL_CERT = os.path.join(BASEPATH, 'cert.pem')
SSL_CERT_CHAIN = None
SSL_PRIVKEY = os.path.join(BASEPATH, 'privkey.pem')

# By default, the server will allow negotiations with extremely old protocols
# that are susceptible to attacks, so we only allow TLSv1.2
class SecuredSSLServer(pyOpenSSLAdapter):
    def get_context(self):
        c = super(SecuredSSLServer, self).get_context()
        c.set_options(SSL.OP_NO_SSLv2)
        c.set_options(SSL.OP_NO_SSLv3)
        c.set_options(SSL.OP_NO_TLSv1)
        c.set_options(SSL.OP_NO_TLSv1_1)
        return c


# Create our own sub-class of Bottle's ServerAdapter
# so that we can specify SSL. Using just server='cherrypy'
# uses the default cherrypy server, which doesn't use SSL
class SSLCherryPyServer(ServerAdapter):
    def run(self, handler):
        cert = SSL_CERT
        #cert_chain = SSL_CERT_CHAIN
        privkey = SSL_PRIVKEY
        server = WSGIServer((self.host, self.port), handler)
        server.ssl_adapter = SecuredSSLServer(cert, privkey) #, cert_chain)
        try:
            server.start()
        finally:
            server.stop()


def pretty_print_time(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%02d:%02d:%02d" % (h, m, s)


def enable_cors(fn):
    """Enable cross origin resource sharing (CORS).

    A decorator to enable CORS for certain handlers.
    """
    def _enable_cors(*args, **kwargs):
        # set CORS headers
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Origin, Accept, Content-Type, X-Requested-With, X-CSRF-Token'

        if request.method != 'OPTIONS':
            # actual request; reply with the actual response
            return fn(*args, **kwargs)

    return _enable_cors
