---
layout: post
title: "Steps to build a github site like this"
date: 2023-05-31 11:42:00 +0800
tags: [githubpages]
---

### full guide by github: 
https://docs.github.com/en/pages/quickstart (quite tedious)

### helpful Chinese guide: 
https://www.zhihu.com/question/20962496 (I basically followed this guide)

### env setup on macos:
```
brew install ruby@2.6
echo 'export PATH="/usr/local/opt/ruby@2.6/bin:$PATH"' >> ~/.zshrc
gem install bundler:2.1.4
bundle install

```
### how to support comments in blogs:
https://utteranc.es/

copy the given script like below to _layouts/post.html below {% raw %}{{ content }}{% endraw %} section

{% raw %}
```
<script src="https://utteranc.es/client.js"
        repo="seanzqs/seanzqs.github.io"
        issue-term="pathname"
        label="comment"
        theme="photon-dark"
        crossorigin="anonymous"
        async>
</script>
```
{% endraw %}





