# Third-party notices

The project's own code is licensed under the Apache License 2.0 (see [LICENSE](LICENSE)).
The following bundled material is third-party and keeps its original license.

## Web UI fonts

`src/fhir_healthcare_ai/api/static/atkinson-next.woff2` and `atkinson-mono.woff2`:
Atkinson Hyperlegible Next and Atkinson Hyperlegible Mono, from the fontsource-variable
packages, unmodified. SIL Open Font License 1.1; the full text ships next to the fonts in
[`FONTS-LICENSE.txt`](src/fhir_healthcare_ai/api/static/FONTS-LICENSE.txt).

## Claude Code skills (`.claude/skills/`)

Development aids for Claude Code, vendored with `npx skills add` and pinned in
[`skills-lock.json`](skills-lock.json). They are not part of the Python package or the
container image.

### emilkowalski/skill

Skills: animate, animate-expo, animation-vocabulary, apple-design, ask-sonner, emil-design-eng, find-animation-opportunities, improve-animations, mobile-native, pick-ui-library, prototype, review-animations, write-swift.
Source: <https://github.com/emilkowalski/skill>

```text
MIT License

Copyright (c) 2026 Emil Kowalski

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### nextlevelbuilder/ui-ux-pro-max-skill

Skill: ui-ux-pro-max.
Source: <https://github.com/nextlevelbuilder/ui-ux-pro-max-skill>

```text
MIT License

Copyright (c) 2024 Next Level Builder

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### anthropics/skills

Skill: frontend-design. Source: <https://github.com/anthropics/skills>. Apache License
2.0; the full text ships in
[`.claude/skills/frontend-design/LICENSE.txt`](.claude/skills/frontend-design/LICENSE.txt).
