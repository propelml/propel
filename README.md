# Propel

A Machine Learning Framework for JavaScript (and TypeScript!)

[![Build Status](https://travis-ci.com/propelml/propel.svg?token=eWz4oGVxypBGsz78gdKp&branch=master)](https://travis-ci.com/propelml/propel)

## Testing

To run the tests, use the following commands:

```bash
# Run all tests in node.js
ts-node test_node

# Run all web browser tests
npm run webpack
ts-node test_browser

# Run specific tests (node.js)
ts-node util_test           # all tests defined in util_test.ts
ts-node test_node mat       # tests that have 'mat' in the name
ts-node test_node "^mnist"  # tests names matching the regex /^mnist/i
```

If puppeteer complains that chromium hasn't been downloaded, make it so:

```bash
npm rebuild puppeteer
```
