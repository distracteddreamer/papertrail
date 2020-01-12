import argparse
import nbconvert
import os
from datetime import datetime

HEADER = """---
layout: post
title:  {}
date:   {}
categories: jekyll update
---"""

parser = argparse.ArgumentParser()

parser.add_argument('--nb_name', type=str,
    required=True, help="Name of notebook")
parser.add_argument('--post_name', type=str,
    required=False, 
    help="Name of post (if not provided will be the same as notebook)")
parser.add_argument('--title', type=str,
    required=True, help="Post title")

args = parser.parse_args()

post_name = args.post_name if args.post_name is not None else args.nb_name

notebook_file = '{}.ipynb'.format(args.nb_name)

exporter = nbconvert.MarkdownExporter()
exporter.preprocessors.append(
    nbconvert.preprocessors.ExtractOutputPreprocessor()
)

print('Exporting notebook {}'.format(notebook_file))
body, resources = exporter.from_file(notebook_file)

if len(resources['outputs']) > 0:
    resources_folder = 'assets/{}'.format(post_name)

    print('Making resources folder {}'.format(resources_folder))
    resources_path = '../' + resources_folder 
    if not os.path.exists(resources_path):
        os.makedirs(resources_path)
        
    for name, img in resources['outputs'].items():
        with open(os.path.join(resources_path, name), 'wb') as f:
            f.write(img)
            print('Saved {}'.format(name))

print('Formatting post')
date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Add header 
post = HEADER.format(args.title, date) + '\n\n' + body

# Add reference to url 
post = post.replace('![png](',
'![png]({{ site.baseurl }}/%s/' % resources_folder)

post_filename = '{}-{}.md'.format(date.split()[0], post_name)
print('Saving post {}'.format(post_filename))
with open('../_posts/{}'.format(post_filename), 'w') as f:
    f.write(post)