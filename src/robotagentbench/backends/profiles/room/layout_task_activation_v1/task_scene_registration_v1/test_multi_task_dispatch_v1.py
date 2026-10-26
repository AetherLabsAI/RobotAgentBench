"""CPU policy fixtures only; not native navigation or manipulation success."""
import math
import unittest
from multi_task_dispatch_v1 import MultiTaskDispatcher,TaskRegistration

def task(name,x,y,hz=240):
    return TaskRegistration(name,name,(x,y),((x-.1,y-.1),(x+.1,y-.1),(x+.1,y+.1),(x-.1,y+.1)),hz)

class Tests(unittest.TestCase):
    def make(self,tasks):
        d=MultiTaskDispatcher(tasks,sector_radius=1.32,sector_half_angle=45)
        d.start(accepted_scene=True,simulation_time_s=5)
        return d
    def begin(self,d,yaw=0,t=5,xy=(0,0)):
        return d.select(dict(xy=xy,yaw=yaw),simulation_time_s=t)
    def tick(self,d,yaw=0,t=5,xy=(0,0),score=None):
        selection=self.begin(d,yaw,t,xy)
        if selection['timed_out']:return selection
        if score is not None:d.record_score(selection['current_task'],dict(score=score))
        return dict(selection,**{'outcome':d.finish_frame()})
    def test_front_priority_even_if_farther(self):
        d=self.make([task('front',1,0),task('near',.5,.3)])
        self.assertEqual(self.tick(d)['current_task'],'front')
    def test_angle_tie_distance(self):
        d=self.make([task('far',1,0),task('near',.5,0)])
        self.assertEqual(self.tick(d)['current_task'],'near')
    def test_only_sector_hit_no_exit_retention(self):
        d=self.make([task('coffee',1,0)])
        self.assertEqual(self.tick(d)['current_task'],'coffee')
        self.assertIsNone(self.tick(d,yaw=180,t=5.05)['current_task'])
    def test_independent_score_reentry(self):
        d=self.make([task('coffee',1,0),task('fryer',-1,0)])
        self.tick(d,score=2);self.tick(d,180,6,score=1)
        self.tick(d,0,7)
        self.assertEqual(d.scores['coffee']['score'],2)
        self.assertEqual(d.scores['fryer']['score'],1)
        self.assertEqual(d.last_progress_time,6)
        self.assertEqual(d.started_at,5)
    def test_noncurrent_score_rejected(self):
        d=self.make([task('coffee',1,0),task('fryer',-1,0)])
        self.begin(d)
        with self.assertRaises(ValueError):d.record_score('fryer',dict(score=2))
    def test_no_progress_120_seconds_while_away(self):
        d=self.make([task('coffee',1,0)])
        self.tick(d)
        self.assertEqual(self.tick(d,180,124)['outcome']['remaining_s'],1)
        result=self.tick(d,180,125)['outcome']
        self.assertTrue(result['timed_out'])
        self.assertEqual(result['termination_reason'],'no_score_increase_timeout')
        with self.assertRaises(RuntimeError):self.tick(d,0,126)
    def test_gain_extends_beyond_total_120_seconds(self):
        d=self.make([task('coffee',1,0)])
        self.tick(d,t=124,score=1)
        r=self.tick(d,t=200)['outcome']
        self.assertEqual(r['elapsed_s'],195)
        self.assertFalse(r['timed_out'])
        self.assertEqual(r['remaining_s'],44)
        self.assertTrue(self.tick(d,t=244)['outcome']['timed_out'])
    def test_each_gain_restarts_shared_timer(self):
        d=self.make([task('coffee',1,0),task('fryer',-1,0)])
        for i,t in enumerate((100,200,300,400)):
            self.tick(d,0 if i%2==0 else 180,t,score=1+i//2)
            self.assertEqual(d.last_progress_time,t)
        self.assertEqual(d.total_score,4)
        self.assertEqual(self.tick(d,t=519)['outcome']['remaining_s'],1)
    def test_switch_never_counts_existing_score_as_gain(self):
        d=self.make([task('coffee',1,0),task('fryer',-1,0)])
        self.tick(d,t=6,score=5);self.tick(d,180,7,score=1)
        self.tick(d,t=100,score=5)
        self.assertEqual(d.total_score,6)
        self.assertEqual(d.last_progress_time,7)
        self.assertTrue(self.tick(d,180,127,score=1)['outcome']['timed_out'])
    def test_same_score_and_deduction_do_not_extend(self):
        d=self.make([task('coffee',1,0)])
        self.tick(d,t=10,score=2)
        self.tick(d,t=20,score=2)
        self.tick(d,t=30,score=1)
        self.assertEqual(d.last_progress_time,10)
        self.assertTrue(self.tick(d,t=130)['outcome']['timed_out'])
    def test_actual_increase_after_deduction_counts(self):
        # Literal net-score increase, not an unapproved historical-high rule.
        d=self.make([task('coffee',1,0)])
        self.tick(d,t=10,score=2);self.tick(d,t=20,score=1);self.tick(d,t=30,score=2)
        self.assertEqual(d.last_progress_time,30)
    def test_exact_deadline_gain_precedes_timeout(self):
        d=self.make([task('coffee',1,0)])
        result=self.tick(d,t=125,score=1)['outcome']
        self.assertFalse(result['timed_out'])
        self.assertEqual(result['remaining_s'],120)
    def test_late_gain_cannot_revive(self):
        d=self.make([task('coffee',1,0)])
        self.assertTrue(self.begin(d,t=125.05)['timed_out'])
        with self.assertRaises(RuntimeError):d.record_score('coffee',dict(score=1))
    def test_deadline_20hz_float_accumulation(self):
        d=self.make([task('coffee',1,0)])
        now=5
        for i in range(2400):
            now+=.05
            r=self.tick(d,t=now)['outcome']
            self.assertEqual(r['timed_out'],i==2399)
    def test_duplicate_score_in_frame_rejected(self):
        d=self.make([task('coffee',1,0)]);self.begin(d)
        d.record_score('coffee',dict(score=1))
        with self.assertRaises(RuntimeError):d.record_score('coffee',dict(score=2))
    def test_score_outside_frame_rejected(self):
        d=self.make([task('coffee',1,0)])
        with self.assertRaises(RuntimeError):d.record_score('coffee',dict(score=1))
    def test_incomplete_or_repeated_frame_rejected(self):
        d=self.make([task('coffee',1,0)]);self.begin(d)
        with self.assertRaises(RuntimeError):self.begin(d,t=6)
        d.finish_frame()
        with self.assertRaises(ValueError):self.begin(d,t=5)
    def test_current_task_hz(self):
        d=self.make([task('coffee',1,0),task('table',-1,0,60)])
        self.assertEqual(self.tick(d)['requested_physics_hz'],240)
        self.assertEqual(self.tick(d,180,6)['requested_physics_hz'],60)
        self.assertEqual(self.tick(d,90,7)['requested_physics_hz'],60)
    def test_zero_one_many(self):
        self.assertIsNone(self.tick(self.make([]))['current_task'])
        for n in (1,2,7):
            d=self.make([task(str(i),1,i*.01) for i in range(n)])
            self.assertEqual(len(self.tick(d)['candidates']),n)
    def test_duplicate_and_unapproved_priority(self):
        with self.assertRaises(ValueError):self.make([task('a',1,0),task('a',2,0)])
        with self.assertRaises(ValueError):
            MultiTaskDispatcher([],sector_radius=1.32,sector_half_angle=45,selection_order='distance_then_front')
    def test_start_once(self):
        d=self.make([])
        with self.assertRaises(RuntimeError):d.start(accepted_scene=True,simulation_time_s=5)
    def test_exact_tie_stable_id(self):
        d=self.make([task('b',1,0),task('a',1,0)])
        self.assertEqual(self.tick(d)['current_task'],'a')
    def test_invalid_score_transactional(self):
        d=self.make([task('coffee',1,0)]);self.begin(d)
        for v in (True,math.nan,math.inf,'3'):
            with self.assertRaises(ValueError):d.record_score('coffee',dict(score=v))
        self.assertEqual(d.total_score,0)
        self.assertEqual(d.last_progress_time,5)
        self.assertFalse(d.scored)
    def test_external_mutation_cannot_fake_progress(self):
        d=self.make([task('coffee',1,0)]);self.begin(d)
        result=dict(score=2,details=dict(phase='first'))
        d.record_score('coffee',result);d.finish_frame()
        result['score']=100;result['details']['phase']='changed'
        copy=d.scores;copy['coffee']['score']=50
        self.assertEqual(d.total_score,2)
        self.assertEqual(d.scores['coffee']['details']['phase'],'first')
    def test_no_unapproved_fixed_total_timeout(self):
        with self.assertRaises(TypeError):
            MultiTaskDispatcher([],sector_radius=1.32,sector_half_angle=45,total_timeout_s=120)
    def test_configurable_inactivity_value(self):
        d=MultiTaskDispatcher([],sector_radius=1.32,sector_half_angle=45,no_progress_timeout_s=60)
        d.start(accepted_scene=True,simulation_time_s=0)
        d.select(dict(xy=(0,0),yaw=0),simulation_time_s=60)
        self.assertTrue(d.finish_frame()['timed_out'])
    def test_invalid_inactivity_value(self):
        for value in (0,-1,True,math.nan,math.inf):
            with self.assertRaises(ValueError):
                MultiTaskDispatcher([],sector_radius=1.32,sector_half_angle=45,no_progress_timeout_s=value)

    def test_native_frame_end_timestamp(self):
        d=self.make([task('coffee',1,0)]);self.begin(d)
        d.complete_physics_frame(simulation_time_s=5.05)
        d.record_score('coffee',dict(score=1));d.finish_frame()
        self.assertEqual(d.last_progress_time,5.05)
    def test_native_wrong_interval_rejected(self):
        d=self.make([]);self.begin(d)
        with self.assertRaises((ValueError,RuntimeError)):d.complete_physics_frame(simulation_time_s=5.1)
    def test_native_deadline_gain(self):
        d=self.make([task('coffee',1,0)]);self.begin(d,t=124.95)
        d.complete_physics_frame(simulation_time_s=125)
        d.record_score('coffee',dict(score=1))
        self.assertFalse(d.finish_frame()['timed_out'])

if __name__=='__main__':unittest.main()
