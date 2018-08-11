# bmanohar's blog
### Personal Blog on Data Science and Artificial Intelligence

[![Build Status](https://travis-ci.org/bmanohar16/blog.svg?branch=master)](https://travis-ci.org/bmanohar16/blog)
[![Ruby](https://img.shields.io/badge/ruby-2.5.1-blue.svg?style=flat)](http://travis-ci.org/bmanohar16/blog)
[![Jekyll](https://img.shields.io/badge/jekyll-3.8.3-red.svg?style=flat)](http://travis-ci.org/bmanohar16/blog)

## Getting Started

### Deployment

**Important:**  For security reasons, Github does not allow plugins (under `_plugins/`) when deploying with Github Pages. 

I built the site with [travis-ci](https://travis-ci.org/) automatically pushing the
generated HTML files to a *gh-pages* branch.

You will need to set up travis-ci for your repository. Briefly all you
need then is to change your details in *[\_config.yml](_config.yml)* so that you can push to your github repo. You will also need to generate a secure key to add to your
*[.travis.yml](.travis.yml)*
Use the travis gem to generate a secure key with `travis encrypt 'GIT_NAME="YOUR_USERNAME" GIT_EMAIL="YOUR_EMAIL" GH_TOKEN=YOUR_TOKEN'` and using [GitHub Token](https://github.com/settings/tokens)

This approach has clear advantages in that you simply push your file changes to GitHub and all the HTML files are generated for you and pushed to *gh-pages*.

### Compiling Styles

CSS styles are compiled using Gulp/PostCSS to polyfill future CSS spec. You'll need Node and Gulp installed globally. After that, from the theme's root directory:

```bash
$ npm install
$ gulp
```

Now you can edit `/assets/css/` files, which will be compiled to `/assets/built/` automatically.

## Issues and Contributing

If you run into any problems, please log them on the [issue tracker](https://github.com/bmanohar16/blog/issues).


## Copyright & License

Copyright (C) 2018 - Released under the MIT License.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.