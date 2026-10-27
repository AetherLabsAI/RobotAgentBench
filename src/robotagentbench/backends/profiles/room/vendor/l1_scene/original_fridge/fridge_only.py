"""L1 refrigerator-only source extracted from the archived MW Studio server.

No robot initialization, microwave controls, playback class or HTTP server.
Bodies are retained as AST-selected L1 statements; the approved contact-release
change remains in arena_api.l1_fridge. This file is not a standalone service.
"""

def define_door_material():
    _dmat = stage.DefinePrim('/World/PhysicsMaterials/DoorMat', 'Material')
    _dmat.ApplyAPI(UsdPhysics.MaterialAPI)
    UsdPhysics.MaterialAPI(_dmat).CreateStaticFrictionAttr().Set(2.5)
    UsdPhysics.MaterialAPI(_dmat).CreateDynamicFrictionAttr().Set(2.2)
    PhysxSchema.PhysxMaterialAPI.Apply(_dmat).CreateFrictionCombineModeAttr().Set('max')

def initialize_fridge():
    FRIDGE_A = '/World/Background/Shell/RefrigeratorRecovered'
    FRIDGE_DAMP = float(os.environ.get('FRIDGE_DAMP', 0.6))
    FRIDGE_FRIC_V = float(os.environ.get('FRIDGE_LATCH_FRICTION', 0.05))
    FRIDGE_DOOR_MASS = float(os.environ.get('FRIDGE_DOOR_MASS', 2.0))
    FR_LATCH = {'zone': 'latch', 'deg': float(os.environ.get('FR_LATCH_DEG', 8.0)), 'relatch': float(os.environ.get('FR_RELATCH_DEG', 1.5)), 'free_fric': float(os.environ.get('FR_FREE_FRIC', 0.001)), 'free_damp': float(os.environ.get('FR_FREE_DAMP', 0.002)), 'open_stiff': float(os.environ.get('FR_OPEN_STIFF', 0.0)), 'speed': float(os.environ.get('FR_AUTO_SPEED', 55.0)), 'stop': float(os.environ.get('FR_AUTO_STOP', 89.0)), 'close_trig': float(os.environ.get('FR_CLOSE_TRIG', 10.0)), 'auto_damp': float(os.environ.get('FR_AUTO_DAMP', 120.0)), 'auto_force': float(os.environ.get('FR_AUTO_FORCE', 2000.0)), 'cmd': 0.0, 'run': 55.0, 'prev_th': None, 'hit': 0, 'unlatched': False, 'trig': float(os.environ.get('FR_SPEED_TRIG', 25.0)), 'latch_grip': float(os.environ.get('FR_LATCH_GRIP', 3.0))}
    FR_HINGE = {}
    FR_RCOL = []
    FR_KEEP_COL = tuple((x for x in os.environ.get('FR_KEEP_COL', 'col_handle').split(',') if x))
    FR_DOOR_PATH = f'{FRIDGE_A}/right_door'
    FR_FACE = {}
    FR_FACE_DOT = float(os.environ.get('FR_FACE_DOT', 0.5))
    FR_CT = {'sub': None, 'ev': 0, 'ok': False, 'err': '', 'flag': False, 'age': 999, 'nout': 0, 'nin': 0, 'log': [], 'hold': int(os.environ.get('FR_OUT_HOLD', 10))}
    if PHYS and stage.GetPrimAtPath(FRIDGE_A):
        _fa = stage.GetPrimAtPath(FRIDGE_A)
        _fa.RemoveAPI(UsdPhysics.ArticulationRootAPI)
        _LKL = os.environ.get('FR_LOCK_LEFT', '0') == '1'
        for (_fp, _fk) in ((f'{FRIDGE_A}/body', True), (f'{FRIDGE_A}/left_door', _LKL), (f'{FRIDGE_A}/right_door', False)):
            _pr = stage.GetPrimAtPath(_fp)
            if _pr:
                _pr.CreateAttribute('physics:rigidBodyEnabled', Sdf.ValueTypeNames.Bool).Set(True)
                _pr.CreateAttribute('physics:kinematicEnabled', Sdf.ValueTypeNames.Bool).Set(_fk)
                if not _fk:
                    _pr.CreateAttribute('physics:mass', Sdf.ValueTypeNames.Float).Set(FRIDGE_DOOR_MASS)
                    _pxfd = PhysxSchema.PhysxRigidBodyAPI.Apply(_pr)
                    _pxfd.CreateSolverPositionIterationCountAttr().Set(16)
                    _pxfd.CreateSolverVelocityIterationCountAttr().Set(4)
                    _pxfd.CreateMaxDepenetrationVelocityAttr().Set(1.0)
                    _pxfd.CreateSleepThresholdAttr().Set(0.0)
                    _pxfd.CreateStabilizationThresholdAttr().Set(0.0)
                    _pxfd.CreateMaxAngularVelocityAttr().Set(50.0)
                    _pxfd.CreateAngularDampingAttr().Set(float(os.environ.get('FRIDGE_ANG_DAMP', 0.0)))
                    _pxfd.CreateLinearDampingAttr().Set(0.0)
                    print(f"[studio] 冰箱门 {_fp.split('/')[-1]}: 角阻尼={_pxfd.GetAngularDampingAttr().Get()} 休眠阈值={_pxfd.GetSleepThresholdAttr().Get()}", flush=True)
        FR_RCOL[:] = [_c for _c in stage.GetPrimAtPath(f'{FRIDGE_A}/right_door').GetChildren() if _c.GetTypeName() == 'Cube' and _c.IsActive()]
        print(f'[studio] 右门碰撞体 {len(FR_RCOL)} 个, 机关期间保留 {FR_KEEP_COL}', flush=True)
        FR_DOOR_PATH = f'{FRIDGE_A}/right_door'
        FR_FACE = {}
        for _c in stage.GetPrimAtPath(FR_DOOR_PATH).GetChildren():
            if not _c.HasAPI(UsdPhysics.CollisionAPI):
                continue
            _nmF = _c.GetName()
            try:
                _czF = float(UsdGeom.Xformable(_c).GetLocalTransformation().ExtractTranslation()[2])
            except Exception:
                _czF = 0.0
            FR_FACE[_nmF] = ('out' if 'handle' in _nmF else 'face' if _nmF in ('col_door', 'col_panel') else 'in', _czF)
        _lipdy = float(os.environ.get('MW_LIP_DY', '0.010'))
        if abs(_lipdy) > 1e-06:
            _nlip = 0
            for _lp in Usd.PrimRange(stage.GetPrimAtPath(f'{FRIDGE_A}/right_door')):
                if _lp.GetName() not in ('col_shelf1_lip', 'col_shelf2_lip', 'col_shelf3_lip', 'col_shelf4_lip'):
                    continue
                _xa = _lp.GetAttribute('xformOp:translate')
                _sa = _lp.GetAttribute('xformOp:scale')
                if not _xa or not _sa:
                    continue
                _t = [float(v) for v in _xa.Get()]
                _s = [float(v) for v in _sa.Get()]
                _ytop0 = _t[1] + _s[1]
                _t[1] += _lipdy / 2.0
                _s[1] += _lipdy / 2.0
                _xa.Set(Gf.Vec3d(*_t))
                _sa.Set(Gf.Vec3f(*[float(v) for v in _s]))
                _vp = UsdGeom.Cube.Define(stage, f'{_lp.GetPath().pathString}_visext')
                _vp.CreateSizeAttr().Set(2.0)
                _vx = UsdGeom.Xformable(_vp.GetPrim())
                _vx.ClearXformOpOrder()
                _vx.AddTranslateOp().Set(Gf.Vec3d(_t[0], _ytop0 + _lipdy / 2.0, _t[2]))
                _vx.AddScaleOp().Set(Gf.Vec3f(float(_s[0]), float(_lipdy / 2.0), float(_s[2])))
                _vp.CreateDisplayColorAttr().Set([(0.72, 0.74, 0.76)])
                _vp.GetPrim().CreateAttribute('physics:collisionEnabled', Sdf.ValueTypeNames.Bool).Set(False)
                _nlip += 1
            print(f'[studio] 栏杆加高 {100 * _lipdy:.1f}cm: {_nlip} 件 (露出底板 2.5 -> {2.5 + 100 * _lipdy:.1f}cm; 碰撞体, 可见网格未动)', flush=True)
        _crdF = PhysxSchema.PhysxContactReportAPI.Apply(stage.GetPrimAtPath(FR_DOOR_PATH))
        _crdF.CreateThresholdAttr().Set(0.0)
        print('[studio] 门面判据(局部z): ' + ' '.join((f'{k}={v[0]}@{v[1]:.3f}' for (k, v) in sorted(FR_FACE.items()))), flush=True)
        for _hn in ('left_door_hinge', 'right_door_hinge'):
            _hp = stage.GetPrimAtPath(f'{FRIDGE_A}/{_hn}')
            if not _hp:
                continue
            _hp.SetActive(True)
            _hd = UsdPhysics.DriveAPI.Apply(_hp, 'angular')
            _hd.CreateStiffnessAttr().Set(0.0)
            _hd.CreateDampingAttr().Set(FRIDGE_DAMP)
            _hd.CreateTargetPositionAttr().Set(0.0)
            _hd.CreateTargetVelocityAttr().Set(0.0)
            _hf = PhysxSchema.PhysxJointAPI.Apply(_hp).CreateJointFrictionAttr()
            _hf.Set(FRIDGE_FRIC_V)
            FR_HINGE[_hn] = (_hd, _hf)
        for _dn in ('left_door', 'right_door'):
            for _cn in ('col_door', 'col_handle', 'col_panel', 'col_bin34_floor', 'col_bin34_rail', 'col_bin34_panel', 'col_bin34_endR', 'col_shelf4_panel', 'col_shelf4_rail'):
                _cp = stage.GetPrimAtPath(f'{FRIDGE_A}/{_dn}/{_cn}')
                if _cp:
                    _cp.CreateRelationship('material:binding:physics', False).SetTargets([Sdf.Path('/World/PhysicsMaterials/DoorMat')])
        for _dn2 in ('body', 'left_door', 'right_door'):
            _dp2 = stage.GetPrimAtPath(f'{FRIDGE_A}/{_dn2}')
            if not _dp2:
                continue
            for _cc in Usd.PrimRange(_dp2):
                if not _cc.HasAPI(UsdPhysics.CollisionAPI):
                    continue
                _pxc2 = PhysxSchema.PhysxCollisionAPI.Apply(_cc)
                _pxc2.CreateContactOffsetAttr().Set(0.002)
                _pxc2.CreateRestOffsetAttr().Set(0.0)
        print(f'[studio] 左门锁死={_LKL}', flush=True)
        print(f'[studio] 冰箱门 dynamic: damp={FRIDGE_DAMP} 摩擦={FRIDGE_FRIC_V} 门{FRIDGE_DOOR_MASS}kg', flush=True)

def _fridge_pose(which='right'):
    """右/左门 physx 真值: (位置, 旋转矩阵, 开角°) 开角0=关, 正=开"""
    from omni.physx import get_physx_interface as _gpiF
    _r = _gpiF().get_rigidbody_transformation(f'{FRIDGE_ROOT_A}/{which}_door')
    if not _r.get('ret_val'):
        return (None, None, None)
    _p = np.array([float(v) for v in _r['position']])
    (x, y, z, w) = [float(v) for v in _r['rotation']]
    _R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    _ax = _R[:, 0]
    _th = math.degrees(math.atan2(_ax[1], _ax[0])) - FRIDGE_YAW0
    return (_p, _R, (_th + 180) % 360 - 180)

def _quat_mul(a, b):
    (w1, x1, y1, z1) = a
    (w2, x2, y2, z2) = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])

def _quat_inv(q):
    return np.array([q[0], -q[1], -q[2], -q[3]]) / float(np.dot(q, q))

def _quat_rot(q, v):
    return _quat_mul(_quat_mul(q, np.array([0.0, v[0], v[1], v[2]])), _quat_inv(q))[1:]

def _ref_pose_wxyz(_path):
    from omni.physx import get_physx_interface as _gpiDP
    _rq = _gpiDP().get_rigidbody_transformation(_path)
    if not _rq.get('ret_val'):
        return (None, None)
    _p = np.array([float(v) for v in _rq['position']])
    _x = [float(v) for v in _rq['rotation']]
    return (_p, np.array([_x[3], _x[0], _x[1], _x[2]]))

def _door_pose_wxyz():
    return _ref_pose_wxyz(f'{FRIDGE_A}/right_door')

def _docks_step():
    """通用钉门跟随: 把注册在 DOCKS 里的每个物体按记录的门局部位姿摆到位(幽灵态)。
    和 TRAY_DOCK 同机制, 只是支持多个 —— 三选一场景里格子里有多个目标物。"""
    if not DOCKS:
        return
    try:
        (_dp, _dq) = _ref_pose_wxyz(f'{FRIDGE_A}/right_door')
        if _dp is None:
            return
    except Exception:
        return
    for _d in DOCKS:
        try:
            if not _d.get('on', True):
                continue
            if _d.get('ghost', True):
                for _cp in Usd.PrimRange(stage.GetPrimAtPath(_d['root'])):
                    if not _cp.HasAPI(UsdPhysics.CollisionAPI):
                        continue
                    _ca = _cp.GetAttribute('physics:collisionEnabled')
                    if _ca and _ca.Get():
                        _ca.Set(False)
                _ga = stage.GetPrimAtPath(_d['root'] + '/Asset').GetAttribute('physxRigidBody:disableGravity')
                if _ga and (not _ga.Get()):
                    _ga.Set(True)
            _wp = _dp + _quat_rot(_dq, np.asarray(_d['lp'], dtype=float))
            _wq = _quat_mul(_dq, np.asarray(_d['lq'], dtype=float))
            if _d.get('rp') is None:
                _d['rp'] = SingleRigidPrim(_d['root'] + '/Asset')
                _d['rp'].initialize()
            _d['rp'].set_world_pose(_wp, _wq)
            _d['rp'].set_linear_velocity(np.zeros(3))
            _d['rp'].set_angular_velocity(np.zeros(3))
        except Exception:
            _d['rp'] = None

def _tray_dock_step():
    """盘钉门跟随一步: 幽灵态盘随右门格子走 (开门/关门/机关全程)。抓取前 tray_drop2/settle 释放。"""
    _docks_step()
    if not TRAY_DOCK.get('on'):
        return
    try:
        (_dp, _dq) = _ref_pose_wxyz(TRAY_DOCK.get('refp') or f'{FRIDGE_A}/right_door')
        if _dp is None:
            return
        _cbF = stage.GetPrimAtPath(TRAY_ROOT + '/Asset/col_box')
        if _cbF:
            _ceF = _cbF.GetAttribute('physics:collisionEnabled')
            if _ceF and _ceF.Get():
                _ceF.Set(False)
        _dgF = tray_asset.GetAttribute('physxRigidBody:disableGravity')
        if _dgF and (not _dgF.Get()):
            _dgF.Set(True)
        _wp = _dp + _quat_rot(_dq, TRAY_DOCK['lp'])
        _wq = _quat_mul(_dq, TRAY_DOCK['lq'])
        _rp = TRAY_DOCK.get('rp')
        if _rp is None:
            _rp = SingleRigidPrim(TRAY_ROOT + '/Asset')
            _rp.initialize()
            TRAY_DOCK['rp'] = _rp
        _rp.set_world_pose(_wp, _wq)
        _rp.set_linear_velocity(np.zeros(3))
        _rp.set_angular_velocity(np.zeros(3))
    except Exception as _eTD:
        TRAY_DOCK['rp'] = None
        if TRAY_DOCK.get('_err') != repr(_eTD):
            TRAY_DOCK['_err'] = repr(_eTD)
            print(f'[tray_dock] 跟随异常(每种只报一次): {_eTD!r}', flush=True)

def _fr_on_contact(_hs, _cd):
    """PhysX 接触上报回调: 判定这一物理步里右门是被**外侧**还是内侧碰到。
    每步由物理引擎直接调用(不经主循环), 只做归类计数, 不动任何驱动参数。"""
    FR_CT['ev'] += 1
    if FR_LATCH.get('zone') != 'open':
        return
    try:
        _xfC = None
        _w2l = None
        for _h in _hs:
            _a0 = str(PhysicsSchemaTools.intToSdfPath(_h.actor0))
            _a1 = str(PhysicsSchemaTools.intToSdfPath(_h.actor1))
            if _a0 == FR_DOOR_PATH:
                _cdr = str(PhysicsSchemaTools.intToSdfPath(_h.collider0))
                _oth = _a1
            elif _a1 == FR_DOOR_PATH:
                _cdr = str(PhysicsSchemaTools.intToSdfPath(_h.collider1))
                _oth = _a0
            else:
                continue
            if _oth.startswith(FRIDGE_A):
                continue
            (_kd, _czC) = FR_FACE.get(_cdr.rsplit('/', 1)[-1], ('in', 0.0))
            if _w2l is None:
                _xfC = UsdGeom.XformCache()
                _l2w = _xfC.GetLocalToWorldTransform(stage.GetPrimAtPath(FR_DOOR_PATH))
                _w2l = _l2w.GetInverse()
                _nw = _l2w.TransformDir(Gf.Vec3d(0, 0, 1))
                _nw = _nw / max(1e-09, float(Gf.Vec3d(_nw).GetLength()))
            for _i in range(_h.contact_data_offset, _h.contact_data_offset + _h.num_contact_data):
                _ci = _cd[_i]
                _pp = _ci.position
                try:
                    _pw = Gf.Vec3d(float(_pp.x), float(_pp.y), float(_pp.z))
                except Exception:
                    _pw = Gf.Vec3d(*[float(_v) for _v in _pp])
                _lz = float(_w2l.Transform(_pw)[2])
                _nn = _ci.normal
                try:
                    _nv = Gf.Vec3d(float(_nn.x), float(_nn.y), float(_nn.z))
                except Exception:
                    _nv = Gf.Vec3d(*[float(_v) for _v in _nn])
                _face = abs(float(_nv[0] * _nw[0] + _nv[1] * _nw[1] + _nv[2] * _nw[2])) > FR_FACE_DOT
                _isout = _kd == 'out' or (_kd == 'face' and _lz > _czC and _face)
                if _isout:
                    FR_CT['flag'] = True
                    FR_CT['nout'] += 1
                else:
                    FR_CT['nin'] += 1
                FR_CT['log'].append(f"{_cdr.rsplit('/', 1)[-1]}({_kd}) z={_lz:.3f} 面{('✓' if _face else '✗')} {('外' if _isout else '内')} <- {_oth.rsplit('/', 1)[-1]}")
                if len(FR_CT['log']) > 60:
                    del FR_CT['log'][:-60]
    except Exception as _ec:
        FR_CT['err'] = repr(_ec)

def _fr_lcol(on):
    """开/关左门碰撞(左门幽灵态). 左门关着不动, 幽灵让右门储物格开门时能穿过它,
    消掉'储物格扫过门缝顶住左门'那个卡点(用户 08-09 定的思路)。"""
    for _cn in ('col_panel', 'col_door'):
        _p = stage.GetPrimAtPath(f'{FRIDGE_A}/left_door/{_cn}')
        if _p:
            _a = _p.GetAttribute('physics:collisionEnabled')
            _a.Set(bool(on)) if _a else _p.CreateAttribute('physics:collisionEnabled', Sdf.ValueTypeNames.Bool).Set(bool(on))

def _fr_boxbody_filter(on):
    """开/关门**摆动期**过滤 盒×柜体 碰撞对(FilteredPairs)。
    真物理盒的盒尾在门摆动时要扫过柜腔, 勾住柜体就卡门(实测卡 0.8-18°,
    fr_test 无真盒 8/8 全开 —— 单因子隔离+插桩定位, 手位姿两边完全一致)。
    钉盒时代摆动期盒子是幽灵态, 从未暴露。门格碰撞不动(盒子照常被门托着随门转),
    latch/open 态恢复完整接触(取盒物理不受影响)。与"左门幽灵"(_fr_lcol)同族手术。"""
    from pxr import UsdPhysics as _UPf
    _body = stage.GetPrimAtPath(f'{FRIDGE_A}/body')
    if not _body:
        return
    _ldoor = stage.GetPrimAtPath(f'{FRIDGE_A}/left_door')
    _tgts = [p.GetPath() for p in (_body, _ldoor) if p and p.IsValid()]
    for _rt in ('/World/Assets/BeefTray', '/World/Assets/SaladBoxR', '/World/Assets/BinFoodR'):
        _as = stage.GetPrimAtPath(_rt + '/Asset')
        if not _as or not _as.IsValid():
            continue
        _api = _UPf.FilteredPairsAPI.Apply(_as)
        _rel = _api.GetFilteredPairsRel()
        _rel.SetTargets(_tgts if on else [])

def _fr_latch_simple():
    """★干净的开关门驱动(simple_latch=1): 去掉破封/关门机关/重咬合/沉降那套互相打架的
    历史状态机 —— 它会在门开到 30 多度自己误触发'破封 -> 自由摆'把驱动断掉(实测停角飘忽
    20/33/35/39°)。这里只有四态、纯位置驱动:
      latch  : 门闩摩擦保持; 手勾把手拉出速度 > trig -> opening(并让左门幽灵)
      opening: 目标角匀速升到 stop(89°), 位置驱动稳稳开到位 -> open
      open   : 锁定 89°(取盒时盒子蹭门也不带关); 被明显外力推(速度<-trig 连续2帧) -> closing
      closing: 目标角匀速降到 0 -> latch(恢复左门碰撞、重新咬合)
    右门碰撞全程不关(open_keepcol 真物理), 把手是贴合可见的网格。"""
    if not (PHYS and FR_HINGE):
        return
    _tray_dock_step()
    try:
        _th = _fridge_pose('right')[2]
        if _th is None:
            return
        (_hd, _hf) = FR_HINGE['right_door_hinge']
        _pv = FR_LATCH.get('prev_th')
        _w = 0.0 if _pv is None else (_th - _pv) * 60.0
        FR_LATCH['prev_th'] = _th
        _z = FR_LATCH.get('zone', 'latch')
        _trig = FR_LATCH['trig']
        _stop = FR_LATCH['stop']
        _spd = FR_LATCH['speed']

        def _pos_drive(tgt):
            _hf.Set(0.0)
            _hd.CreateTargetVelocityAttr().Set(0.0)
            _hd.GetStiffnessAttr().Set(FR_LATCH['auto_damp'])
            _hd.GetDampingAttr().Set(FR_LATCH['auto_damp'] * 0.15)
            _hd.CreateMaxForceAttr().Set(FR_LATCH['auto_force'])
            _hd.GetTargetPositionAttr().Set(float(tgt))
        if _z == 'latch':
            _hf.Set(FR_LATCH.get('latch_grip', FRIDGE_FRIC_V))
            _hd.GetStiffnessAttr().Set(0.0)
            _hd.GetDampingAttr().Set(FRIDGE_DAMP)
            _hd.GetTargetPositionAttr().Set(0.0)
            FR_LATCH['hit'] = FR_LATCH.get('hit', 0) + 1 if _w > _trig and _th > 0.5 else 0
            if FR_LATCH['hit'] >= 2:
                FR_LATCH['zone'] = 'opening'
                FR_LATCH['cmd'] = float(_th)
                FR_LATCH['hit'] = 0
                _fr_lcol(False)
                print(f'[fridge] 简化开门触发@{_th:.1f}° ({_w:.0f}°/s) -> 匀速到 {_stop:.0f}° (左门幽灵)', flush=True)
        elif _z == 'opening':
            FR_LATCH['cmd'] = min(_stop, FR_LATCH.get('cmd', _th) + _spd / 60.0)
            _pos_drive(FR_LATCH['cmd'])
            if _th >= _stop - 0.5:
                FR_LATCH['zone'] = 'open'
                _fr_boxbody_filter(False)
                print(f'[fridge] 简化开门到位 {_th:.1f}° (盒×柜体/左门 碰撞已恢复)', flush=True)
        elif _z == 'open':
            _pos_drive(_stop)
            FR_LATCH['chit'] = FR_LATCH.get('chit', 0) + 1 if _w < -_trig else 0
            if FR_LATCH['chit'] >= 2:
                FR_LATCH['zone'] = 'closing'
                FR_LATCH['cmd'] = float(_th)
                FR_LATCH['chit'] = 0
                _fr_lcol(False)
                _fr_boxbody_filter(True)
                print(f'[fridge] 简化关门触发@{_th:.1f}° ({-_w:.0f}°/s)', flush=True)
        elif _z == 'closing':
            FR_LATCH['cmd'] = max(0.0, FR_LATCH.get('cmd', _th) - _spd / 60.0)
            _pos_drive(FR_LATCH['cmd'])
            if _th <= 0.5:
                FR_LATCH['zone'] = 'latch'
                FR_LATCH['cmd'] = 0.0
                _hd.GetStiffnessAttr().Set(0.0)
                _hd.GetTargetPositionAttr().Set(0.0)
                _hd.GetDampingAttr().Set(FRIDGE_DAMP)
                _hf.Set(FRIDGE_FRIC_V)
                _fr_lcol(True)
                print(f'[fridge] 简化关门到位, 重新咬合', flush=True)
    except Exception as _e:
        print(f'[fridge] simple_latch 异常: {_e}', flush=True)

def _fr_latch_step():
    if FR_TUNE.get('simple_latch', 0.0) >= 0.5:
        return _fr_latch_simple()
    '冰箱门机关一步(速度触发匀速开/关门)。主循环和 fr_test 都调它。'
    if not (PHYS and FR_HINGE):
        return
    if not FR_CT['ok'] and FR_CT['ev'] > 0:
        FR_CT['ok'] = True
        print(f"[fridge] 接触上报可用 (事件{FR_CT['ev']}) 判据就绪; 是否用它管住门: out_only={FR_TUNE.get('out_only', 0.0)}", flush=True)
    try:
        _thF = _fridge_pose('right')[2]
        if _thF is not None:
            (_hdF, _hfF) = FR_HINGE['right_door_hinge']
            _pv = FR_LATCH.get('prev_th')
            _w = 0.0 if _pv is None else (_thF - _pv) * 60.0
            FR_LATCH['prev_th'] = _thF
            _zF = FR_LATCH['zone']
            _rcol = _fr_rcol

            def _ramp(v):
                _hfF.Set(FR_LATCH['free_fric'])
                _hdF.CreateTargetVelocityAttr().Set(0.0)
                FR_LATCH['cmd'] = float(np.clip(FR_LATCH['cmd'] + v / 60.0, 0.0, FR_LATCH['stop']))
                _hdF.GetStiffnessAttr().Set(FR_LATCH['auto_damp'])
                _hdF.GetDampingAttr().Set(FR_LATCH['auto_damp'] * 0.15)
                _hdF.CreateMaxForceAttr().Set(FR_LATCH['auto_force'])
                _hdF.GetTargetPositionAttr().Set(FR_LATCH['cmd'])
            if FR_LATCH.pop('rcol_go', None):
                _rcol(True)
                print('[fridge] 手已撤离 -> 门碰撞恢复', flush=True)
            if FR_LATCH.get('settle', 0) > 0:
                FR_LATCH['settle'] -= 1
                _hfF.Set(FRIDGE_FRIC_V)
                _hdF.CreateTargetVelocityAttr().Set(0.0)
                _hdF.GetTargetPositionAttr().Set(0.0)
                _hdF.GetStiffnessAttr().Set(float(FR_LATCH.get('settle_stiff', 2000.0)))
                _hdF.GetDampingAttr().Set(float(FR_LATCH.get('settle_damp', 600.0)))
                FR_LATCH['cmd'] = 0.0
                FR_LATCH['hit'] = 0
                if FR_LATCH['settle'] == 0:
                    _hdF.GetStiffnessAttr().Set(0.0)
                    _hdF.GetDampingAttr().Set(FRIDGE_DAMP)
                    print(f'[fridge] 门闩沉降完成 @{_thF:.1f}° (残余动量已吸收)', flush=True)
                _tray_dock_step()
                return
            _TRG = FR_LATCH['trig']
            if _zF == 'latch' and _thF > FR_LATCH['deg'] and (not FR_LATCH.get('unlatched')):
                _hfF.Set(FR_LATCH['free_fric'])
                _hdF.GetDampingAttr().Set(FR_LATCH['free_damp'])
                FR_LATCH['unlatched'] = True
                print(f"[fridge] 破封@{_thF:.1f}° -> 自由摆 (摩擦{FR_LATCH['free_fric']})", flush=True)
            if _zF == 'latch':
                FR_LATCH['hit'] = FR_LATCH.get('hit', 0) + 1 if _w > _TRG and _thF > 1.0 else 0
                if FR_LATCH['hit'] >= 2:
                    FR_LATCH['cmd'] = float(_thF)
                    FR_LATCH['run'] = max(_w, FR_LATCH['speed'])
                    if FR_TUNE.get('open_keepcol', 0.0) < 0.5:
                        _rcol(False, hold_inner=False)
                    _ramp(FR_LATCH['run'])
                    FR_LATCH['zone'] = 'opening'
                    print(f"[fridge] 机关触发@{_thF:.1f}° (实测 {_w:.0f}°/s) -> 碰撞已关, 锁定 {FR_LATCH['run']:.0f}°/s 开到 {FR_LATCH['stop']:.0f}°", flush=True)
            elif _zF == 'opening':
                if _thF >= FR_LATCH['stop'] - 0.5:
                    FR_LATCH['cmd'] = FR_LATCH['stop']
                    _ramp(0.0)
                    FR_LATCH['zone'] = 'open'
                    _rcol(True)
                    print(f'[fridge] 已开到位 {_thF:.1f}° (碰撞已恢复)', flush=True)
                else:
                    _ramp(FR_LATCH['run'])
            elif _zF == 'open':
                if FR_CT['flag']:
                    FR_CT['age'] = 0
                    FR_CT['flag'] = False
                else:
                    FR_CT['age'] += 1
                _out_push = FR_CT['age'] <= FR_CT['hold'] if FR_CT['ok'] and FR_TUNE.get('out_only', 1.0) >= 0.5 else True
                FR_CT['t'] = FR_CT.get('t', 0) + 1
                if _out_push and (not FR_CT.get('was_out')):
                    print(f"[fridge] 外侧接触 -> 门可推 | {(FR_CT['log'][-1] if FR_CT['log'] else '(无接触数据)')}", flush=True)
                elif not _out_push and FR_CT['nin'] > FR_CT.get('nin_p', 0) and (FR_CT['t'] - FR_CT.get('t_p', -999) > 90):
                    FR_CT['t_p'] = FR_CT['t']
                    print(f"[fridge] 内侧接触 门保持不动 ({FR_CT['nin'] - FR_CT.get('nin_p', 0)}点) | {(FR_CT['log'][-1] if FR_CT['log'] else '')}", flush=True)
                    FR_CT['nin_p'] = FR_CT['nin']
                FR_CT['was_out'] = _out_push
                if not _out_push:
                    _ramp(0.0)
                if _w < -_TRG and _out_push:
                    FR_LATCH['cmd'] = float(_thF)
                    FR_LATCH['run'] = max(-_w, FR_LATCH['speed'])
                    if FR_TUNE.get('open_keepcol', 0.0) < 0.5:
                        _rcol(False)
                    _ramp(-FR_LATCH['run'])
                    FR_LATCH['zone'] = 'closing'
                    print(f'[fridge] 关门机关触发@{_thF:.1f}° ({-_w:.0f}°/s)', flush=True)
                elif _out_push:
                    _hfF.Set(float(FR_LATCH.get('open_hold_fric', os.environ.get('FR_OPEN_HOLD_FRIC', 0.3))))
                    _hdF.GetStiffnessAttr().Set(0.0)
                    _hdF.GetDampingAttr().Set(float(FR_LATCH.get('open_hold_damp', os.environ.get('FR_OPEN_HOLD_DAMP', 0.05))))
                    FR_LATCH['cmd'] = float(_thF)
            elif _zF == 'closing':
                if _thF <= FR_LATCH['relatch']:
                    _hdF.CreateTargetVelocityAttr().Set(0.0)
                    _hdF.GetStiffnessAttr().Set(0.0)
                    _hdF.GetTargetPositionAttr().Set(0.0)
                    _hdF.GetDampingAttr().Set(FRIDGE_DAMP)
                    _hfF.Set(FRIDGE_FRIC_V)
                    FR_LATCH['zone'] = 'latch'
                    FR_LATCH['unlatched'] = False
                    FR_LATCH['settle'] = int(FR_LATCH.get('settle_steps', 45))
                    FR_LATCH['rcol_go'] = False
                    print(f'[fridge] 关到位@{_thF:.1f}° 重新咬合 (碰撞待撤手后恢复)', flush=True)
                else:
                    _ramp(-FR_LATCH['run'])
    except Exception:
        pass
    _tray_dock_step()

def _fr_rcol(on, hold_inner=None):
    """右门碰撞总开关。开/关门机关期间临时关掉, 免得被手臂卡住(实测手臂在初始位姿时
    门的上限只有 80.8°, 复位关门也会被从另一侧顶住)。
    ★FR_KEEP_COL 里的部件始终保留碰撞 —— 默认保留最外层把手, 这样视觉上手和把手仍有
    真实接触, 只让门板/储物格穿过去(实际会顶到手的本来也是门板)。

    ★★08-08 修"关门时陪衬盒下落": 关门机关触发那一行也调 _rcol(False), 于是**储物格
    连同门板一起失去碰撞** —— 盒子脚下的托板瞬间消失, 直接穿下去。实测三条成功集
    六个陪衬盒**全部**在 `close 关门` 那一刻坠落(z 0.817 -> -36.5 -> 末帧 -2008m,
    还有落到地板 0.122 的), 而且都已经录进训练数据。
    修法: 内侧件(储物格/搁板/挡杆/挡板)**始终保留碰撞**。依据是 FR_FACE 那套按资产
    几何算出来的 in/face/out 分类 —— 手臂只接触 face(门板)和 out(把手), 这正是
    "门只能从外侧推动"判据的前提, 所以保住 in 不会把手臂卡住。
    注意别把 out 一起保: FR_KEEP_COL 默认的 col_handle 正是手要抓的地方, 保它就是
    当年 keep_col=1 让门卡在 22.5° 的原因。
    出问题可在线关: `fr_tune hold_inner_col 0`。

    ★★08-08(同日再修, 实测): 上面这条**不能用在开门段**。开门机关那一行也调 _rcol(False),
    保住内侧件之后整排储物格带着碰撞扫过去, 门**卡死在 41.9°**(实测三次一致; 关掉
    hold_inner_col 立刻开到 88.8°)。储物格自由边角离铰链 47.8cm, 比门板外角的 44.9cm
    还远 2.9cm —— 门板过得去的地方它过不去。
    而"陪衬盒穿地掉落"只发生在**关门**段(盒子那时静静躺在格子里, 托板一消失就穿下去),
    开门段盒子本来就是钉住/幽灵态, 不需要托板。所以按段分开:
      hold_inner=False -> 开门机关 / 手动驱动(要的是"谁也挡不住门")
      hold_inner=True  -> 关门机关 / _fridge_close_doors(要的是"盒子脚下有托板")

    ★★08-08 open_keepcol(用户: "机关保留, 开关门不要幽灵态, 好让盒子不用钉门"):
      开着(>=0.5)时**忽略一切关碰撞请求**(not on 直接返回), 门碰撞恒全开。
      机关照常匀速开/关(auto_force=10^5 能把储物格挤过离左门最紧的 2.4cm@33°),
      盒子靠真物理坐格子里跟门转, 不再 tray_dock/DOCKS 幽灵态。
      默认 0 = 现状不变(安全); 在线 `fr_tune open_keepcol 1` 启用, 存盘。
      注意: 启用后**必须同时让盒子物理化**(别再钉门) —— 幽灵盘每帧被传送等效无限质量,
            碰撞一开会反把门顶开(见 _tray_dock_step 注释)。
      ★短路**只**加在 _fr_latch_step 的机关开/关门两处; 复位(_fridge_close_doors)/手动
        fridge_drive 这类脚手架仍可关碰撞 —— 否则复位要穿过 f0 手臂把门关回 0° 会被挡在
        62.8°(实测), 后续开门从 62.8° 起步等于没测到。
    ★★08-09 修正上面那条"只加在机关两处": 短路**提到入口**(用户"取消冰箱幽灵态")。
      复位(_fridge_close_doors)那条路仍会关掉整扇门**含格子底板**, 配 hold_inner_col=0
      -> 摆盒瞬间托板消失, 实测 `[bin] ⚠盒子掉穿(门局部y -85.3 << 目标 0.830)`,
      三盒 release 后全 z=0 躺地板 -> 取盒必 ✗(盒离爪口 123cm)。
      当初怕"复位关门被 f0 手臂挡在 62.8°", 但那个真凶已查明是 DebugBinSupport cube(已删),
      删后纯铰链全碰撞开关门 5/5 到 90°, 顾虑不再成立。
    """
    if not on and FR_TUNE.get('open_keepcol', 0.0) >= 0.5:
        return
    for _cp in FR_RCOL:
        _keep = FR_KEEP_COL if FR_TUNE.get('keep_col', 0.0) >= 0.5 else ()
        _hi = FR_TUNE.get('hold_inner_col', 1.0) >= 0.5 if hold_inner is None else bool(hold_inner)
        _hold = _hi and FR_FACE.get(_cp.GetName(), ('', 0.0))[0] == 'in'
        _v = True if not on and (_cp.GetName() in _keep or _hold) else bool(on)
        _a = _cp.GetAttribute('physics:collisionEnabled')
        _a.Set(_v) if _a else _cp.CreateAttribute('physics:collisionEnabled', Sdf.ValueTypeNames.Bool).Set(_v)

def tune_fridge():
    for (_k, _v) in c.items():
        if _k != 'cmd':
            FR_TUNE[_k] = float(_v)
    for (_k9, _n9) in (('latch_deg', 'deg'), ('relatch_deg', 'relatch'), ('open_stiff', 'open_stiff'), ('free_fric', 'free_fric'), ('free_damp', 'free_damp'), ('auto_speed', 'speed'), ('auto_stop', 'stop'), ('close_trig', 'close_trig'), ('auto_damp', 'auto_damp'), ('auto_force', 'auto_force'), ('speed_trig', 'trig'), ('latch_grip', 'latch_grip'), ('open_hold_fric', 'open_hold_fric'), ('open_hold_damp', 'open_hold_damp')):
        if _k9 in FR_TUNE:
            FR_LATCH[_n9] = float(FR_TUNE[_k9])
