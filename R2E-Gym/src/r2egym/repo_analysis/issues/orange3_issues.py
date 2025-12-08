orange3_issues = [
    r"""Currently, this code will be served to the browser as text/plain thus the HTML are not rendered by the browser:

from wsgiref.simple_server import make_server
from pyramid.config import Configurator

def hello_world(request):
    request.response.content_type = "text/html"
    return "<p>Hello World</p>"

config = Configurator()
config.add_route('hello', '/')
config.add_view(hello_world, route_name='hello', renderer='string')
app = config.make_wsgi_app()
make_server('', 8000, app).serve_forever()
A little bit of investigative work shows that the issue is here:

File: /pyramid/renderers.py
155 def string_renderer_factory(info):
156     def _render(value, system):
157         if not isinstance(value, string_types):
158             value = str(value)
159         request = system.get('request')
160         if request is not None:
161             response = request.response
162             ct = response.content_type
163             if ct == response.default_content_type:
164                 response.content_type = 'text/plain'
165         return value
166     return _render
Here response.default_content_type == 'text/html' and the string renderer replaces the specified content_type with its default of text/plain. I think this is unintuitive/unexpected behavior, instead when request.response.content_type is explicitly set to 'text/html', the renderer should be not change it.

I didn't test the json renderer, but since there is a similar code over there, I'd assume it has the same bug as well.
"""
]
