import assert from 'node:assert/strict';
import {angleArc, normalizeCoordinateInput} from '../src/contact_angle/static/app.js';

const baseline = [{x:10,y:100},{x:500,y:100}];
const contact = {x:150,y:135}, radius = 24, degrees = 124;
const direction = radians => ({x:Math.cos(radians), y:Math.sin(radians)});
for (const side of ['left','right']) {
  for (const pendant of [true,false]) {
    const tilt = pendant ? .27 : 0;
    const base = side==='left' ? tilt : Math.PI-tilt;
    const sweep = degrees*Math.PI/180*(side==='left'?1:-1)*(pendant?1:-1);
    const tangent = direction(base+sweep);
    const geometry = pendant ? {surface_tangent:direction(base),tangent_vector:tangent} : {};
    const arc = angleArc(contact,baseline,side,degrees,radius,36,geometry);
    for (const [p,theta] of [[arc[0],base],[arc[18],base+sweep/2],[arc[36],base+sweep]]) {
      assert.ok(Math.hypot(p.x-contact.x-radius*Math.cos(theta),p.y-contact.y-radius*Math.sin(theta))<1e-8);
    }
  }
}
const imported = normalizeCoordinateInput({orientation:'pendant',baseline,baseline_source:'automatic',
  baseline_points:[{x:55,y:80},{x:90,y:60},{x:420,y:60},{x:450,y:80}],
  baseline_method:'pendant_surface_v1'},640,480);
assert.equal(imported.orientation,'pendant');
assert.equal(imported.baseline_method,'pendant_surface_v1');
assert.deepEqual(imported.baseline,baseline);
assert.throws(()=>normalizeCoordinateInput({orientation:'upside',baseline},640,480));
