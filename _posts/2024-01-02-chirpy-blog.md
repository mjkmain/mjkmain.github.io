---
title: Github 블로그 만드는 법
author: mjkmain
date: 2024-01-02 00:43:00 +0900
categories: [Blogging]
tags: [typography]
pin: true
math: true
render_with_liquid: false
---

# [Chirpy 테마](https://github.com/cotes2020/jekyll-theme-chirpy)로 github 블로그 구축하기

3일 걸렸습니다. 여러분은 30분 걸리시길 바랍니다. [(공식 가이드)](https://chirpy.cotes.page/posts/getting-started/)

[Chirpy starter](https://github.com/cotes2020/chirpy-starter)를 사용하는 방법과 [Github Fork](https://github.com/cotes2020/jekyll-theme-chirpy/fork)를 통해 구축하는 방법이 있는데, Chirpy starter로 시도해본 결과 5분이면 만들 수 있지만, 여러가지 customizing이 어렵다는 점이 마음에 안들어서 Github Fork를 통해 진행했습니다.

## Environment 
- OS : Ubuntu 22.04.3 LTS
- Ruby : 3.2.0 
- Bundler : 2.5.3
- npm : 10.2.4
- nodejs : v12.22.9

## Prerequisites
- [해당 사이트](https://vegastack.com/tutorials/how-to-install-ruby-on-rails-with-rbenv-on-ubuntu-22-04/)를 참고해서 rbenv 설치 후 `rbenv install 3.2.0` 명령어를 통해 ruby 3.2.0 다운로드
- `rbenv global 3.2.0` 명령어로 3.2.0 version을 default version으로 적용
- `ruby -v`로 버전 확인
```console
ruby -v
> ruby 3.2.0 (2022-12-25 revision a528908271) [x86_64-linux]
```
- node.js, npm 설치
```console
$ sudo apt install nodejs
$ sudo apt install npm
$ nodejc -v
> v12.22.9
$ npm -v
> 10.2.4
```

## Process
### 1. Fork & Clone
[Github Fork](https://github.com/cotes2020/jekyll-theme-chirpy/fork)를 통해 fork를 진행합니다. 이때, repository name은 아래 사진과 같이 `<username>.github.io`로 지정합니다.


![Desktop View](https://github.com/mjkmain/blog-image/assets/72269271/338cc4fb-e33e-4612-b823-c3285cf5ccf4){: width="680" } 
<!-- fork.png -->

다음으로 git clone을 진행합니다.
```console
git clone https://github.com/<username>/<username>.github.io
cd <username>.github.io
```

### 2. Initialization
`tools/init` script를 통해 initialize를 진행합니다. 블로그 root 디렉토리 (`<username>.github.io`{: .filepath})에서 `bash tools/init` 명령어를 실행합니다.

아래처럼 Initialization succesful 메시지가 보이면 성공적으로 된거고, 이게 안뜨고 여러 에러가 뜨면 [**prerequisites**](#prerequisites)가 모두 올바르게 설치되었는지 확인이 필요합니다.

```bash
mjkim@mlp-server:~/mjkmain.github.io$ bash tools/init 
HEAD is now at 60836af chore(release): 6.3.1
npm WARN deprecated @babel/plugin-proposal-class-properties@7.18.6: This proposal has been merged to the ECMAScript standard and thus this plugin is no longer maintained. Please use @babel/plugin-transform-class-properties instead.

added 400 packages, and audited 401 packages in 21s

67 packages are looking for funding
  run `npm fund` for details

found 0 vulnerabilities

> jekyll-theme-chirpy@6.3.1 prebuild
> npx rimraf assets/js/dist


> jekyll-theme-chirpy@6.3.1 build
> NODE_ENV=production npx rollup -c --bundleConfigAsCjs


_javascript/commons.js → assets/js/dist/commons.min.js...
created assets/js/dist/commons.min.js in 1s

_javascript/home.js → assets/js/dist/home.min.js...
created assets/js/dist/home.min.js in 636ms

_javascript/categories.js → assets/js/dist/categories.min.js...
created assets/js/dist/categories.min.js in 596ms

_javascript/page.js → assets/js/dist/page.min.js...
created assets/js/dist/page.min.js in 606ms

_javascript/post.js → assets/js/dist/post.min.js...
created assets/js/dist/post.min.js in 588ms

_javascript/misc.js → assets/js/dist/misc.min.js...
created assets/js/dist/misc.min.js in 532ms

[INFO] Initialization successful!
```

### 3. Installing Dependency
`bundle` 명령어를 통해 dependency를 맞춰줍니다.

```bash
mjkim@mlp-server:~/mjkmain.github.io$ bundle
Fetching gem metadata from https://rubygems.org/..........
Resolving dependencies...
Bundle complete! 6 Gemfile dependencies, 47 gems now installed.
Use `bundle info [gemname]` to see where a bundled gem is installed.
1 installed gem you directly depend on is looking for funding.
  Run `bundle fund` for details
```

### 4. ruby version 
[해당 사이트](https://talk.jekyllrb.com/t/build-error-at-setup-ruby-stage-of-build-and-deploy-on-actions/8782)를 참고했습니다.

`.github/workflows/pages-deploy.yml`{: .filepath} 파일에서 `ruby-version`을 수정해야 합니다.

![Desktop View](https://github.com/mjkmain/blog-image/assets/72269271/7de9b2cd-05a7-427c-adb9-32f40cfec442){: width="400"}
<!-- ruby-version.png -->

이 작업을 하지 않으면 github action에서 아래와 같은 에러가 발생합니다. 
```console
An error occurred while installing google-protobuf (3.25.1), and Bundler cannot
continue.

In Gemfile:
  jekyll-theme-chirpy was resolved to 6.3.1, which depends on
    jekyll-archives was resolved to 2.2.1, which depends on
      jekyll was resolved to 4.3.2, which depends on
        jekyll-sass-converter was resolved to 3.0.0, which depends on
          sass-embedded was resolved to 1.69.5, which depends on
            google-protobuf
Error: The process '/opt/hostedtoolcache/Ruby/3.3.0/x64/bin/bundle' failed with exit code 5
```

### 5. Customization
`_config.yml`{: .filepath}에서 `url`, `timezone`을 수정해줍니다. 

- `url`: `https://<username>.github.io` 
- `timezone` : `Asia/Seoul`

이 외에 바꿀만한게 많은데, 알아서 변경하시면 될 것 같습니다.

### 6. Github setting
깃헙 레포지토리로 가서 Settings-Pages에서 `Build and deployment`를 Github Actions로 변경해야합니다. 

![Desktop View](https://github.com/mjkmain/blog-image/assets/72269271/2efdaa57-bdc1-4134-8753-fdc26a2c7699){: width="700"}
<!-- action-setting.png -->

### 7. Git commit & push
이제 _config.yml 변경된 사항을 add하고, `bash tools/init` 진행 시 발생한 변경사항을 push 합니다. 

`bash tools/init` 명령어를 진행하면 자동적으로 새로운 commit을 만들도록 해놨다고 합니다. 

```console
git add .
git commit -m "commit message"
git push -f
```

`git push`에 `-f` 안붙여주면 아래와 같이 에러났습니다. (이유는 모르겠음)
```bash
mjkim@mlp-server:~/mjkmain.github.io$ git push
To https://github.com/mjkmain/mjkmain.github.io.git
 ! [rejected]        master -> master (non-fast-forward)
error: failed to push some refs to 'https://github.com/mjkmain/mjkmain.github.io.git'
hint: Updates were rejected because the tip of your current branch is behind
hint: its remote counterpart. Integrate the remote changes (e.g.
hint: 'git pull ...') before pushing again.
hint: See the 'Note about fast-forwards' in 'git push --help' for details.
```

블로그 테마 만든 cotes2020아저씨한테 물어보니 그냥 `git push -f` 하라고 하더라고요. [링크](https://github.com/cotes2020/jekyll-theme-chirpy/discussions/1443)

### 8. Action build 확인
아래 사진처럼 action에 초록색 체크 뜨는지 확인하시고 `<username>.github.io` 로 접속해보시면 됩니다. 

![Desktop View](https://github.com/mjkmain/blog-image/assets/72269271/b5d4382d-7c21-44ca-bc3f-06394b037001){: width="600"}
<!-- action.png -->

저는 action run failed 떠서 3일동안 우울했어요 
이메일 오는 소리 PTSD옵니다..
![Desktop View](https://github.com/mjkmain/blog-image/assets/72269271/9c078dea-0b3d-4c48-9d0f-cd0f41a83c77){: width="800"}
<!-- action-failed.png -->

모두들 30분컷 하시길 바랍니다. 아 이미지 왜 안뜨냐