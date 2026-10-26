'use strict';
// A process bridge only, no second implementation of the geometry rules.
const fs=require('node:fs'),G=require('./placement_sources/random-placement-geometry.js');
const q=JSON.parse(fs.readFileSync(0,'utf8'));
const reason=G.checkFootprints(q.placement,q.layout);
process.stdout.write(JSON.stringify({passed:reason===null,reason,kernel:'random-placement-geometry.checkFootprints.v2'}));
