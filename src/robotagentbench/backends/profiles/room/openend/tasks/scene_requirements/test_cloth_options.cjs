const assert=require('node:assert/strict');
const {clothItemRequest}=require('./web/cloth-options.js');
assert.deepEqual(clothItemRequest('random',''),{});
assert.deepEqual(clothItemRequest('random','3'),{count:3});
assert.deepEqual(clothItemRequest('books',''),{items:[{kind:'book_red',count:1}]});
for(let n=1;n<=4;n++)assert.deepEqual(clothItemRequest('books',String(n)),{items:[{kind:'book_red',count:n}]});
for(const n of ['0','5','1.5','foo'])assert.throws(()=>clothItemRequest('books',n));
assert.throws(()=>clothItemRequest('unknown','2'));
console.log('cloth request UI contract passed');
