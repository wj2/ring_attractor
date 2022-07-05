
import numpy as np
import scipy.signal as sps
import itertools as it
import matplotlib.pyplot as plt
import scipy.linalg as scl
import scipy.optimize as spo
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker

import general.plotting as gpl
import general.utility as u

def diff_exponential_weightfunc(diff, a, k1, k2, shift=0):
    weight = (a*np.exp(k1*(np.cos(diff + shift) - 1))
              - a*np.exp(k2*(np.cos(diff + shift) - 1)))
    return weight

def von_mises_weightfunc(diff, a, k, shift=0):
    weight = a*(np.exp((np.cos(diff + shift) - 1)/k) - 1)
    return weight

def von_mises_weightfunc_normed(diff, a, k, shift=0):
    weight = a*(np.exp((np.cos(diff + shift) - 1)/k) - 1)
    norm = 1 - np.exp(-2*k)
    return weight/norm

def cosine_weightfunc(diff, j0, j1, shift=0):
    weight = j0 + j1*np.cos(diff + shift)
    return weight 

def cosine_power_weightfunc(diff, amp, wid, shift=0):
    prec = 1/wid
    weight = amp*(1/(2**prec)*(1 + np.cos(diff + shift))**prec - 1)
    return weight

def exponential_transfer(g, a):
    r = a*np.exp(g)
    return r

def relu_transfer(g, a, thr=0):
    g[g < thr] = 0
    r = a*g
    return r

def tanh_transfer(g, a, b):
    r = a*(1 + np.tanh(g + b))
    return r

def valid_theta(theta):
    return u.normalize_periodic_range(theta)

def norm_pop_traj(traj):
    nt = traj - np.expand_dims(np.mean(traj, 1), 1)
    fnt = nt/np.expand_dims(np.sum(nt**2, 1), 1)
    return fnt

def plot_integ_results(outmap, wids, shifts, delt_bs, ax=None, legend=False,
                       inds=None, plot_wid=False):
    if ax is None:
        f = plt.figure()
        ax = f.add_subplot(1,1,1)
    for i, wid in enumerate(wids):
        for j, shift in enumerate(shifts):
            om = outmap[i, j]
            label = r'$\phi = '+'{:0.2f}\pi$'.format(shift/np.pi)
            if plot_wid:
                label = r'$k = '+'{:0.2f}$'.format(wid)
            om_med = np.median(om, axis=1)
            l = ax.plot(delt_bs, om_med, label=label)
            if inds is not None: 
                lin_xs = (delt_bs[inds[0]], delt_bs[inds[1]])
                lin_om = (om_med[inds[0]], om_med[inds[1]])
                ax.plot(lin_xs, lin_om, '--', color=l[0].get_color())
            for k in range(om.shape[1]):
                ax.plot(delt_bs, om[:, k], 'o', alpha=.1,
                        color=l[0].get_color())
    ax.set_xlabel(r'$\Delta_{b}/b_{0}$')
    ax.set_ylabel('velocity (deg/s)')
    if legend:
        ax.legend(frameon=False)
    return ax

def find_min_params(intmap, decmap, velmap, wids, shifts, lambdas, v_thrs,
                    big_weight=10**10):
    min_wids = []
    min_shifts = []
    min_rfs = []
    min_integs = []
    for i, l in enumerate(lambdas):
        comp = intmap + l*decmap
        l_wids = []
        l_shifts = []
        l_rfs = []
        l_int = []
        for j, vt in enumerate(v_thrs):
            mask = velmap < vt
            weight = mask*big_weight
            comp = comp + weight
            
            mv = np.nanmin(comp)
            m_wids, m_shifts = np.where(comp == mv)
            l_wids.append(wids[m_wids])
            l_shifts.append(shifts[m_shifts])
            l_rfs.append(decmap[m_wids, m_shifts])
            l_int.append(intmap[m_wids, m_shifts])
        min_wids.append(l_wids)
        min_shifts.append(l_shifts)
        min_rfs.append(l_rfs)
        min_integs.append(l_int)
    return min_wids, min_shifts, min_rfs, min_integs

def plot_min_params(min_wids, min_shifts, min_rfs, min_integs, lams, v_thrs,
                    cmap='Blues', start_col_samp=.4, end_col_samp=1,
                    fsize=(5, 4.5)):
    f = plt.figure(figsize=fsize)
    ax_wids = f.add_subplot(4, 1, 1)
    ax_shifts = f.add_subplot(4, 1, 2, sharex=ax_wids)
    ax_rfs = f.add_subplot(4, 1, 3, sharex=ax_wids)
    ax_ints = f.add_subplot(4, 1, 4, sharex=ax_wids)
    cm = mpl.cm.get_cmap(cmap)
    cols = np.linspace(start_col_samp, end_col_samp, len(lams))
    for i, l in enumerate(lams):
        mw_arr = np.array(min_wids[i])
        ms_arr = np.array(min_shifts[i])
        mr_arr = np.array(min_rfs[i])
        mi_arr = np.array(min_integs[i])
        ax_wids.plot(v_thrs, mw_arr, 'o', label='$\lambda = '+'{}$'.format(l),
                     color=cm(cols[i]))
        ax_shifts.plot(v_thrs, ms_arr, 'o', color=cm(cols[i]))
        ax_rfs.plot(v_thrs, mr_arr, 'o', color=cm(cols[i]))
        ax_ints.plot(v_thrs, mi_arr, 'o', color=cm(cols[i]))
    ax_wids.legend(frameon=False)
    ax_wids.set_yscale('log')
    ax_ints.set_yscale('log')
    max_i = 3
    gpl.clean_plot(ax_shifts, 0, max_i=max_i, horiz=False)
    gpl.clean_plot(ax_wids, 1, max_i=max_i, horiz=False)
    gpl.clean_plot(ax_rfs, 2, max_i=max_i, horiz=False)
    gpl.clean_plot(ax_ints, 3, max_i=max_i, horiz=False)
    ax_shifts.set_ylabel('intraring shift')
    ax_wids.set_ylabel('kernel width')
    ax_rfs.set_ylabel('RF size')
    ax_rfs.set_ylabel('integration error')
    ax_ints.set_xlabel('required integration velocity')
    return ax_wids, ax_shifts, ax_rfs, f

def get_integ_maps(outmap, trs, wids, shifts, delt_bs, delt_inds=(0, -1),
                   eps=.0001, rf_bs_ind=0, t_ind=40, integ_time=400.0,
                   thetas=None, eps_err=100, plot_broken=False):
    if thetas is None:
        fk = list(trs.keys())[0]
        sk = list(trs[fk].keys())[0]
        thetas = get_thetas(trs[fk][sk][0].shape[1])
    plot_om = np.zeros((len(wids), len(shifts)))
    vel = np.zeros_like(plot_om)
    rf_wids = np.zeros_like(plot_om)
    ampls = np.zeros_like(plot_om)
    for i, wid in enumerate(wids):
        for j, shift in enumerate(shifts):
            om = outmap[i, j]
            v_diff = om[delt_inds[1]] - om[delt_inds[0]]
            b_diff = delt_bs[delt_inds[1]] - delt_bs[delt_inds[0]]
            slope = v_diff/b_diff
            sub_bs = np.expand_dims(delt_bs[delt_inds[0]:delt_inds[1]], 1)
            lin = sub_bs*np.expand_dims(slope, 0)
            lin = lin + delt_bs[delt_inds[0]]
            comp_vs = om[delt_inds[0]:delt_inds[1]]
            err = np.sum((comp_vs[1:]/lin[1:] - 1)**2, 0)
            if np.std(slope) > eps_err:
                plot_om[i,j] = np.nan
                vel[i,j] = np.nan
            else:
                plot_om[i, j] = np.median(err)
                vel[i, j] = np.median(om[delt_inds[1]])
            rf_b = delt_bs[rf_bs_ind]
            act_slice = trs[(wid, shift)][(rf_b, integ_time)][0][t_ind]
            ampls[i, j] = np.sum(act_slice)
            act_slice[act_slice < eps] = 0
            pks = sps.argrelmax(act_slice, mode='wrap')[0]
            if (len(pks) > 1 or np.isnan(plot_om[i,j]) or np.max(act_slice)
                - np.min(act_slice) < eps):
                # print('wid {:0.2f}, shift {:0.2f}'.format(wid, shift), slope)
                rf_wids[i, j] = np.nan
                plot_om[i, j] = np.nan
                ampls[i, j] = np.nan
                vel[i, j] = np.nan
                if plot_broken:
                    fpk = plt.figure()
                    axpk = fpk.add_subplot(1,1,1)
                    axpk.plot(thetas, act_slice)
                    axpk.plot(thetas[pks], act_slice[pks], 'o')
                    axpk.set_title('wid = {:0.2f}, phi = {:0.2f}'.format(wid,
                                                                         shift))
            else:
                sl_max, sl_min = np.max(act_slice), np.min(act_slice)
                halfval = (sl_max - sl_min)/2 + sl_min
                pts = sps.argrelmin(np.abs(act_slice - halfval), mode='wrap')[0]
                if len(pts) != 2:
                    rf_wids[i, j] = np.nan
                else:
                    if act_slice[pts[0] + 2] > halfval:
                        cwid = thetas[pts[1]] - thetas[pts[0]]
                    else:
                        cwid = 2*np.pi + thetas[pts[0]] - thetas[pts[1]]
                    rf_wids[i, j] = cwid
    return plot_om, vel, rf_wids, ampls

def plot_integ_maps(plot_om, vel, rf_wids, ampls, wids, shifts, figsize=(16,3)):
    f = plt.figure(figsize=figsize)
    ax1 = f.add_subplot(1,4,1)
    ax2 = f.add_subplot(1,4,3, sharex=ax1, sharey=ax1)
    ax3 = f.add_subplot(1,4,2, sharex=ax1, sharey=ax1)
    ax4 = f.add_subplot(1,4,4, sharex=ax1, sharey=ax1)
    ax1.set_yscale('log')
    shifts_pcm = gpl.pcolormesh_axes(shifts, len(shifts))
    wids_pcm = 10**gpl.pcolormesh_axes(np.log10(wids), len(wids))
    pom_map = ax1.pcolormesh(shifts_pcm, wids_pcm, np.log10(plot_om),
                             shading='flat')
    cb1 = f.colorbar(pom_map, ax=(ax1,))
    cb1.set_label('log integration error')
    vel_map = ax2.pcolormesh(shifts_pcm, wids_pcm, vel, shading='flat')
    cb2 = f.colorbar(vel_map, ax=(ax2,))
    cb2.set_label('velocity (deg/s)')
    rf_map = ax3.pcolormesh(shifts_pcm, wids_pcm, rf_wids, shading='flat')
    cb3 = f.colorbar(rf_map, ax=(ax3,))    
    cb3.set_label('bump width (radians)')
    amp_map = ax4.pcolormesh(shifts_pcm, wids_pcm, np.log10(ampls),
                             shading='flat')
    ax1.set_title('integration error')
    ax3.set_title('$\propto$ readout error')
    ax2.set_title('maximum velocity')

    cb4 = f.colorbar(amp_map, ax=(ax4,))
    cb4.set_label('log sum activity')
    ax1.set_ylabel(r'connectivity width ($k$)')
    # ax3.set_ylabel(r'connectivity width ($k$)')
    ax1.set_xlabel('intraring shift (radians)')
    ax2.set_xlabel('intraring shift (radians)')
    ax3.set_xlabel('intraring shift (radians)')
    return ax1, ax2, ax3, ax4, f

def plot_error_maps(integs, rf_wids, wids, shifts, min_wids, min_shifts, lams,
                    figsize=(8, 2.8), pt1_color=(1, .1, .1),
                    pt2_color=(.1, 1, .1), cmap=None):
    f = plt.figure(figsize=figsize)
    ax1 = f.add_subplot(1,2,1)
    ax2 = f.add_subplot(1,2,2, sharex=ax1, sharey=ax1)
    ax1.set_yscale('log')
    shifts_pcm = gpl.pcolormesh_axes(shifts, len(shifts))
    wids_pcm = 10**gpl.pcolormesh_axes(np.log10(wids), len(wids))
    integ_map = ax1.pcolormesh(shifts_pcm, wids_pcm, np.log10(integs),
                               shading='flat', cmap=cmap)
    cb1 = f.colorbar(integ_map, ax=(ax1,))
    cb1.set_label('log integration error')
    rf_map = ax2.pcolormesh(shifts_pcm, wids_pcm, rf_wids, shading='flat',
                            cmap=cmap)
    cb2 = f.colorbar(rf_map, ax=(ax2,))
    ax1.plot(min_shifts, min_wids, 'o')
    ax2.plot(min_shifts, min_wids, 'o')
    ax1.plot(min_shifts[0], min_wids[0], 'o', color=pt1_color,
             label='integration')
    ax1.plot(min_shifts[-1], min_wids[-1], 'o', color=pt2_color,
             label='readout')
    ax1.legend(frameon=False, loc=4)
    ax2.plot(min_shifts[0], min_wids[0], 'o', color=pt1_color)
    ax2.plot(min_shifts[-1], min_wids[-1], 'o', color=pt2_color)
    cb2.set_label('bump width (FWHM)')
    ax1.set_ylabel(r'connectivity width ($k$)')
    ax1.set_xlabel('intraring shift (radians)')
    ax2.set_xlabel('intraring shift (radians)')
    return ax1, ax2, f

def test_ampl_settings(ampls, *args, **kwargs):
    out_trs = {}
    for i, a in enumerate(ampls):
        kwargs['a_intra'] = a
        kwargs['a_inter'] = a
        maps, trs = test_conn_width(*args, **kwargs)
        out_trs[a] = trs
        map_exp = np.expand_dims(maps, 0)
        if i == 0:
            all_maps = map_exp
        else:
            all_maps = np.concatenate((all_maps, map_exp), axis=0)
    return all_maps, out_trs

def test_input_noise(wid, shift, noise_bs, duration=1000, reps=1000, basal_b=0,
                     t_wait=500, integ_dt=.1, n_neurs=1000,
                     tau=10, g=None, tf=exponential_transfer, noise=True,
                     wf=von_mises_weightfunc, a_intra=1, a_inter=1, big_u=.2,
                     keep=10, dynamics='noiseless', stp=False, tau_u=650,
                     tau_x=1, t_post=50):
    if g is None:
        g = tau
    diff_mat = np.zeros((len(noise_bs), reps, 2))
    tf_params = (g,)
    wf_params = (a_intra, wid)
    inter_wf_params = (a_inter, wid)
    intra_shift_rad = shift
    inter_shift_rad = -shift
    trm = TwoCoupledRings(n_neurs, tau, tf, tf_params, wf, 
                          intra_wf_params=wf_params,
                          intra_shift_rad=intra_shift_rad,
                          inter_wf_params=inter_wf_params, 
                          inter_shift_rad=inter_shift_rad, stp=stp,
                          tau_u=tau_u, tau_x=tau_x, big_u=big_u)
    time_end = t_wait + duration
    for i, nb in enumerate(noise_bs):
        for j in range(reps):
            mag = nb/integ_dt
            df1, _ = bias_drive_function_creator(len(trm.thetas),
                                                 mag, t_wait,
                                                 time_end,
                                                 noise=noise)
            df2, _ = bias_drive_function_creator(len(trm.thetas),
                                                 mag, t_wait,
                                                 time_end,
                                                 noise=noise)
            bias = basal_b/integ_dt            
            out = trm.begin_integration(time_end + t_post, integ_dt,
                                        dynamics=dynamics, df1=df1, df2=df2,
                                        bias=bias, keep=keep)
            r1_out, r2_out, time = out
            tr1, fo1, sd1 = r1_out
            tr2, fo2, sd2 = r2_out
            end = fo1[int(time_end/keep)]
            start = fo1[int(t_wait/keep)]
            diff_mat[i, j] = start, end
    return diff_mat

def plot_pop_trace(time, tr1, tr2, fo1, fo2, d1, d2, thetas, t_delay=200,
                   fsize=(6,9), r1_color=(.625, .781, .781),
                   r2_color=(.781, .625, .781), alpha=.5,
                   neutral_color=(.1, .1, .1), delt_eps=.1, ms=1.5):
    f = plt.figure(figsize=fsize)
    gs = GridSpec(1, 5, figure=f)
    ax_ring1 = f.add_subplot(gs[0, :2])
    ax_ring2 = f.add_subplot(gs[0, 3:], sharex=ax_ring1, sharey=ax_ring1)
    ax_inp = f.add_subplot(gs[0, 2], sharey=ax_ring1)
    gpl.clean_plot(ax_inp, 1)
    gpl.clean_plot(ax_ring2, 2)
    gpl.clean_plot(ax_ring1, 0)

    time = time[t_delay:]
    tr1 = tr1[t_delay:]
    tr2 = tr2[t_delay:]
    fo1 = fo1[t_delay:]
    fo2 = fo2[t_delay:]

    delta_d = np.mean(d2[t_delay:] - d1[t_delay:], axis=1)
    ddn_mask = delta_d <= 0
    ddp_mask = delta_d >= 0
    
    ax_inp.plot(delta_d[ddn_mask], time[ddn_mask], color=r1_color)
    ax_inp.plot(delta_d[ddp_mask], time[ddp_mask], color=r2_color)
    ax_inp.plot(np.zeros_like(time), time, color=neutral_color)
    max_inp = np.max(np.abs(delta_d))
    ax_inp.set_xlim([-max_inp, max_inp])
    
    ax_ring1.pcolormesh(thetas, time, tr1, shading='flat', rasterized=True)
    ax_ring2.pcolormesh(thetas, time, tr2, shading='flat', rasterized=True)
    ax_ring1.plot(fo1, time, 'o', markersize=ms, color=r1_color, alpha=alpha)
    ax_ring2.plot(fo2, time, 'o', markersize=ms, color=r2_color, alpha=alpha)

    ax_ring1.set_ylabel('time (ms)')
    ax_ring1.set_xlabel('neuron tuning (deg)')
    ax_ring2.set_xlabel('neuron tuning (deg)')
    ax_inp.set_xlabel(r'$\Delta_{b}$')
    ax_ring1.set_xlim([0, 2*np.pi])
    ax_ring1.set_ylim([time[0], time[-1]])
    ax_ring1.set_xticks([0, np.pi, 2*np.pi])
    _ = ax_ring1.set_xticklabels(['0', r'$\pi$', r'$2\pi$'])
    return ax_ring1, ax_ring2, ax_inp, f

def plot_input_noise(diff_mat, noise_bs, ax=None):
    if ax is None:
        f = plt.figure()
        ax = f.add_subplot(1,1,1)
    db = np.diff(noise_bs)[0]
    for i, nb in enumerate(noise_bs):
        diff_dist = diff_mat[i]
        od_orig = diff_dist[:, 0] - diff_dist[:, 1]
        od = np.arctan2(np.sin(od_orig), np.cos(od_orig))
        xs = np.ones(len(od))*nb + np.random.randn(len(od))*np.sqrt(.005*db)
        ax.plot(xs, od, 'o')
    return ax

def test_conn_width(wids, shifts, delt_bs, basal_b=0, dt_beg=500, dt_end=1500,
                    dt_n=3, t_wait=500, t_post=50, integ_dt=.1, n_neurs=1000,
                    tau=10, g=None, tf=exponential_transfer, stp=False,
                    wf=von_mises_weightfunc, a_intra=1, a_inter=1,
                    keep=10, dynamics='noiseless', inp_noise=False):
    if g is None:
        g = tau
    integ_ranges = np.zeros((len(wids), len(shifts), len(delt_bs), dt_n))
    tr_store = {}
    tf_params = (g,)
    for i, wid in enumerate(wids):
        wf_params = (a_intra, wid)
        inter_wf_params = (a_inter, wid)
        for j, shift in enumerate(shifts):
            intra_shift_rad = shift
            inter_shift_rad = -shift
            trm = TwoCoupledRings(n_neurs, tau, tf, tf_params, wf, 
                                  intra_wf_params=wf_params,
                                  intra_shift_rad=intra_shift_rad,
                                  inter_wf_params=inter_wf_params, 
                                  inter_shift_rad=inter_shift_rad,
                                  stp=stp)
            out = test_bias_integration(trm, delt_bs, basal_b,
                                        dt_beg, dt_end, dt_n,
                                        t_wait, t_post, integ_dt,
                                        keep=keep, dynamics=dynamics,
                                        inp_noise=inp_noise)
            integ_ranges[i, j] = out[0]
            tr_store[(wid, shift)] = out[1]
    return integ_ranges, tr_store

def test_bias_integration(trm, delt_bs, basal_b=0, dt_beg=500, dt_end=1500,
                          dt_n=3, t_wait=500, t_post=100, integ_dt=.1, keep=10,
                          dynamics='noiseless', inp_noise=False):
    delt_ts = np.linspace(dt_beg, dt_end, dt_n)
    integ_ratio = np.zeros((len(delt_bs), dt_n))
    tr_store = {}
    for i, db in enumerate(delt_bs):
        for j, del_t in enumerate(delt_ts):
            t_total = t_wait + del_t + t_post
            n_neurs = len(trm.thetas)
            df1, integ_val1 = bias_drive_function_creator(n_neurs, db, t_wait,
                                                          t_wait + del_t, 
                                                          noise=inp_noise)
            df2, _ = bias_drive_function_creator(n_neurs, -db, t_wait,
                                                 t_wait + del_t, 
                                                 noise=inp_noise)
            out = trm.begin_integration(t_total, integ_dt, df1=df1, df2=df2,
                                        bias=basal_b, keep=keep,
                                        dynamics=dynamics)
            r1_out, r2_out, time = out
            tr1, fo1, sd1 = r1_out
            tr2, fo2, sd2 = r2_out
            tr_store[(db, del_t)] = (tr1, tr2)
            integ_ratio[i, j] = integ_val1(time, smooth_focus(fo1))
    return integ_ratio, tr_store

def smooth_focus(fo, eps=.5):
    dfo = np.diff(fo)
    big_mask = np.abs(dfo) >= 2*np.pi - eps
    for ind in np.where(big_mask)[0]:
        if dfo[ind] < 0:
            fo[ind+1:] = fo[ind+1:] + 2*np.pi
        else:
            fo[ind+1:] = fo[ind+1:] - 2*np.pi
    return fo

def bias_drive_function_creator(n_neurs, magnitude, time_start, time_end,
                                noise=False, basal_b=0):

    def drive_func(t, curr, dt):
        if t < time_start or t > time_end:
            drive = np.zeros(n_neurs)
        else:
            if noise:
                abs_amt = np.random.poisson(np.abs(magnitude*dt))
                amt = np.sign(magnitude)*abs_amt
            else:
                amt = magnitude*dt 
            drive = np.ones(n_neurs)*amt
        return drive
    
    def integ_eval(ts, foc):
        start_ind = np.argmin(np.abs(ts - time_start))
        end_ind = np.argmin(np.abs(ts - time_end))
        delt_time = ts[end_ind] - ts[start_ind]
        delt_foc = foc[end_ind] - foc[start_ind]
        return delt_foc/delt_time
    
    return drive_func, integ_eval

class CueFunc:

    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def __add__(self, f1):
        def new_f(*args, **kwargs):
            return self.func(*args, **kwargs) + f1(*args, **kwargs)
        return CueFunc(new_f)

    def __sub__(self, f1):
        def new_f(*args, **kwargs):
            return self.func(*args, **kwargs) - f1(*args, **kwargs)
        return CueFunc(new_f)


def point_drive_function_creator(pop_thetas, start_theta, width, magnitude,
                                 velocity, time_start, time_end):
    def drive_func(t, curr, dt):
        if t < time_start or t > time_end:
            drive = np.zeros_like(pop_thetas)
        else:
            c_theta = valid_theta(start_theta + velocity*(t - time_start))
            dm1 = pop_thetas > valid_theta(c_theta - width)
            dm2 = pop_thetas < valid_theta(c_theta + width)
            drive_mask = np.logical_and(dm1, dm2)
            drive = magnitude*drive_mask*dt
        return drive
    return CueFunc(drive_func)
    
def step_drive_function_creator(pop_thetas, targ_theta, width, magnitude,
                                time_start, time_end, pop_mask=None):
    pop_mask_col = np.abs(valid_theta(pop_thetas - targ_theta)) < width
    if pop_mask is not None:
        pop_mask = np.logical_and(pop_mask_col, pop_mask)
    else:
        pop_mask = pop_mask_col
    def drive_func(t, curr, dt):
        if t < time_start or t > time_end:
            drive = np.zeros_like(pop_thetas)
        else:
            drive = magnitude*pop_mask*dt
        return drive
    return CueFunc(drive_func)

def get_thetas(n_neurs, start=0, end=2*np.pi):
    return np.linspace(start, end - 2*np.pi/n_neurs, n_neurs)

def get_steps_and_containers(t, t_end, dt, keep, n_neurs, drive_dim=None,
                             dict_keys=None):
    if drive_dim is None:
        drive_dim = n_neurs
        
    steps_to_end = int(np.ceil((t_end - t)/dt))
    interval = int(np.ceil(keep/dt))
    steps_to_keep = int(np.floor(steps_to_end/interval))

    if dict_keys is None:
        trace = np.zeros((steps_to_keep, n_neurs))
        focus = np.zeros(steps_to_keep)
        time = np.zeros(steps_to_keep)
        store_drive = np.zeros((steps_to_keep, drive_dim))
    else:
        trace = {k:np.zeros((steps_to_keep, n_neurs)) for k in dict_keys}
        focus = {k:np.zeros(steps_to_keep) for k in dict_keys}
        time = {k:np.zeros(steps_to_keep) for k in dict_keys}
        store_drive = {k:np.zeros((steps_to_keep, drive_dim)) for k in dict_keys}
    out = (steps_to_end, steps_to_keep, interval, trace, focus, store_drive,
           time)
    return out

def compute_weight_matrix(thetas, wf, wf_params):
    w = np.identity(len(thetas))*wf(0, *wf_params)
    for i, j in it.combinations(range(len(thetas)), 2):
        diff = thetas[i] - thetas[j]
        weight_ij = wf(diff, *wf_params)
        w[i,j] = weight_ij
        weight_ji = wf(-diff, *wf_params)
        w[j,i] = weight_ji
    return w

def summarize_evs(ev_matrix, db, eps=.001, ft=True):
    evals, evecs = np.linalg.eig(ev_matrix)
    ev_mask = evals > 1 + eps
    evals = evals[ev_mask]
    evecs = evecs[:, ev_mask]
    trans_matrix = evals*evecs
    inp_len = int(evecs.shape[0]/2)
    input_vec = np.concatenate((np.ones(inp_len)*db,
                                np.ones(inp_len)*-db))
    if ft:
        input_vec = np.fft.fft(input_vec)
    trans_input = np.dot(trans_matrix.T, input_vec)
    return trans_matrix, trans_input, input_vec

class TwoCoupledRings(object):

    def __init__(self, n_neurs, tau, transfer_func, tf_params,
                 weight_func=diff_exponential_weightfunc,
                 intra_wf_params=(1, 1, .3), intra_shift_rad=2,
                 inter_wf_params=(1, 1, .3), inter_shift_rad=2,
                 stp=False, big_u=.1, tau_x=150, tau_u=650):
        self.thetas = get_thetas(n_neurs)
        self.intra_shift = intra_shift_rad
        self.inter_shift = inter_shift_rad

        wf1_intra_params = intra_wf_params + (-self.intra_shift,)
        wf2_intra_params = intra_wf_params + (self.intra_shift,)
        rm1 = RingAttractor(n_neurs, tau, transfer_func, tf_params,
                            weight_func=weight_func, wf_params=wf1_intra_params,
                            stp=stp, big_u=big_u, tau_x=tau_x, tau_u=tau_u)
        rm2 = RingAttractor(n_neurs, tau, transfer_func, tf_params,
                            weight_func=weight_func, wf_params=wf2_intra_params,
                            stp=stp, big_u=big_u, tau_x=tau_x, tau_u=tau_u)

        wf12_inter_params = inter_wf_params + (-self.inter_shift,)
        self.w1_to_2 = compute_weight_matrix(self.thetas, weight_func,
                                             wf12_inter_params)
        
        wf21_inter_params = inter_wf_params + (self.inter_shift,)
        self.w2_to_1 = compute_weight_matrix(self.thetas, weight_func,
                                             wf21_inter_params)
        self.ring1 = rm1
        self.ring2 = rm2

    def begin_integration(self, t_end, dt, init_cond=None, init_scale=.01,
                          dynamics='noiseless', df1=None, df2=None,
                          bias=0, keep=10):
        if dynamics == 'noiseless':
            dyn1 = self.ring1.dynamics_noiseless
            dyn2 = self.ring2.dynamics_noiseless
        else:
            dyn1 = self.ring1.dynamics_poisson
            dyn2 = self.ring2.dynamics_poisson
        self.ring1.bias = bias
        self.ring2.bias = bias
        self._curr_time = 0
        self.ring1.initialize_network()
        self.ring2.initialize_network()
        out = self.integrate_until(t_end, dt, dyn1=dyn1, keep=keep,
                                   dyn2=dyn2, df1=df1, df2=df2)
        return out

    def get_linear_evolution_matrix(self, sim_time=600, dt=.1, choose_time=-1):
        out1, out2, _ = self.begin_integration(sim_time, dt)
        tr1, _, _ = out1
        tr2, _, _ = out2
        r1_neurs = self.ring1.n_neurs
        r2_neurs = self.ring2.n_neurs
        ddyn1 = tr1[choose_time]*np.identity(r1_neurs)
        ddyn2 = tr2[choose_time]*np.identity(r2_neurs)
        nt1 = np.identity(r1_neurs)*-1
        nt2 = np.identity(r2_neurs)*-1
        ev1_a = nt1 + np.dot(ddyn1, self.ring1.w)
        ev1_b = np.dot(ddyn1, self.w2_to_1)
        ev2_a = np.dot(ddyn2, self.w1_to_2)
        ev2_b = nt2 + np.dot(ddyn2, self.ring2.w)
        ev1 = np.concatenate((ev1_a, ev1_b), axis=1)
        ev2 = np.concatenate((ev2_a, ev2_b), axis=1)
        ev_matrix = np.concatenate((ev1, ev2), axis=0)
        return ev_matrix, tr1[choose_time], tr2[choose_time]

    def get_left_ev_proj(self, eps=.001, db=.5):
        ev_matrix, t1, t2 = self.get_linear_evolution_matrix()
        evals, l_evecs, r_evecs = scl.eig(ev_matrix, left=True, right=True)
        sort_inds = np.argsort(evals)[::-1]
        evals = evals[sort_inds]
        l_evecs = l_evecs[:, sort_inds]
        r_evecs = r_evecs[:, sort_inds]
        assert np.abs(evals[0]) < eps
        el, er = l_evecs[:, 0], r_evecs[:, 0]
        v_prod = np.dot(el, er)
        el_norm = el/v_prod
        inp_len = len(self.thetas)
        input_vec = np.concatenate((t1*db,
                                    -t2*db))
        normed_ivec = input_vec/np.sqrt(np.sum(input_vec**2))
        mag = np.dot(el, normed_ivec)
        return mag, el
    
    def integrate_until(self, t_end, dt, keep=10, dyn1=None, dyn2=None,
                        df1=None, df2=None):
        if df1 is None:
            df1 = lambda t, curr, dt: 0
        if df2 is None:
            df2 = lambda t, curr, dt: 0
        if dyn1 is None:
            dyn1 = self.ring1.dynamics_noiseless
        if dyn2 is None:
            dyn2 = self.ring2.dynamics_noiseless
        out = get_steps_and_containers(self._curr_time, t_end, dt, keep,
                                       self.ring1.n_neurs)
        steps_to_end, steps_to_keep, interval, tr1, fo1, sd1, time = out
        out = get_steps_and_containers(self._curr_time, t_end, dt, keep,
                                       self.ring2.n_neurs)
        _, _, _, tr2, fo2, sd2, _ = out
        j = 0
        for i in range(steps_to_end):
            r1_state = self.ring1.get_state()
            r2_state = self.ring2.get_state()
            d1 = df1(self._curr_time, r1_state, dt)
            d2 = df2(self._curr_time, r2_state, dt)
            r1_to_r2 = np.dot(self.w1_to_2, r1_state)
            r2_to_r1 = np.dot(self.w2_to_1, r2_state)
            drive1 = d1 + r2_to_r1
            drive2 = d2 + r1_to_r2
            if (i % interval) == 0 and j < steps_to_keep:
                tr1[j] = r1_state
                tr2[j] = r2_state
                fo1[j] = self.ring1.thetas[np.argmax(r1_state)]
                fo2[j] = self.ring2.thetas[np.argmax(r2_state)]
                sd1[j] = d1
                sd2[j] = d2
                time[j] = self._curr_time
                j = j + 1
            self.ring1.iterate_step(drive1, dt, dyn1)
            self.ring2.iterate_step(drive2, dt, dyn2)
            self._curr_time = self._curr_time + dt
        return (tr1, fo1, sd1), (tr2, fo2, sd2), time
            
class RingAttractor(object):
    
    def __init__(self, n_neurs, tau, transfer_func, tf_params,
                 weight_func=diff_exponential_weightfunc,
                 wf_params=(1, 1, .3), stp=False, big_u=.1, tau_x=150,
                 tau_u=650, divide_wm=False, bias=10):
        self.thetas = get_thetas(n_neurs)
        self.n_neurs = n_neurs
        self.w = compute_weight_matrix(self.thetas, weight_func,
                                       wf_params)
        if divide_wm:
            self.w = self.w/n_neurs
        self.tf = lambda x: transfer_func(x, *tf_params)
        self.tau = tau
        self.bias = bias
        self.stp = stp
        self.big_u = big_u
        self.tau_u = tau_u
        self.tau_x = tau_x
        self.wf_params = wf_params
        self.bump_stats = None
        self.noise_mag = None

    def compute_bump_statistics_empirical(self, t_until=200, t_step=1, eps=.1):
        if self.bump_stats is None:
            self.initialize_network(init_scale=.1)
            out, focus, _, ts = self.integrate_until(t_until, t_step,
                                                     dynamics='noiseless',
                                                     drive_func=None)
            delt = np.diff(self.thetas)[0]
            tc = np.sum(out[-1] > eps)*delt/2
            a_star = np.max(out[-1])
            a = a_star/(1 - np.cos(tc))
            c = np.cos(tc)*a

            pb_emp = np.sum(out[-1])
            self.bump_stats = (tc, a, c, pb_emp)
        
        return self.bump_stats

    def compute_pc(self, cue, **kwargs):
        tc, a, c, pb_emp = self.compute_bump_statistics_empirical(self, **kwargs)
        if (c + cue) > a:
            pc = 0
        else:
            tc_cue = np.arccos((c + cue)/a)
            pc = 2*a*(np.sin(tc_cue) - tc_cue*np.cos(tc_cue))
        return pc

    def compute_pc_empirical(self, cue, cue_start=200, cue_dur=400, tstep=1):
        t_until = cue_start + cue_dur
        cue_mask = np.mod(np.arange(len(self.thetas)), 2) == 0
        cue_opp = step_drive_function_creator(self.thetas, 0, 2*np.pi, 
                                              -cue, cue_start,
                                              cue_start + cue_dur,
                                              pop_mask=np.logical_not(cue_mask))
        
        out, focus, _, ts = self.integrate_until(t_until, tstep,
                                                 dynamics='noiseless',
                                                 drive_func=cue_opp)
        pb_emp = np.sum(out[-1, np.logical_not(cue_mask)])
        pc_emp = np.sum(out[-1, cue_mask])
        pt_emp = np.sum(out[-1])

        return pb_emp, pc_emp, pt_emp

    def estimate_noise(self, t_until=600, t_step=1, eps=.1,
                       wait=200):
        if self.noise_mag is None:
            self.initialize_network(init_scale=.1)
            out, focus, _, ts = self.integrate_until(t_until, t_step,
                                                     dynamics='poisson',
                                                     drive_func=None)
            delt = np.diff(self.thetas)[0]
            t_mask = ts > wait
            pb_traj = np.sum(out[t_mask], axis=1)*delt
            pb_std = np.std(pb_traj, axis=0)
            
            self.noise_mag = pb_std
        
        return self.noise_mag        
        
    def compute_bump_statistics(self):
        ji, je, _ = self.wf_params
        je = je
        ji = ji
        
        f_tc = lambda tc: (np.pi*2/je + .5*np.sin(2*tc) - tc)**2
        res = spo.minimize(f_tc, .01)
        tc = res.x
        
        denom = -(ji/np.pi)*(np.sin(tc) - tc*np.cos(tc)) - np.cos(tc)
        a = self.bias/denom
        c = a*np.cos(tc)
        return tc, a, c

    def _step_stp_dynamics(self, dt):
        du_1 = -(self.u - self.big_u)/self.tau_u
        du_2 = self.big_u*(1 - self.u)*self._curr_state
        du = du_1 + du_2
        dx = -(self.x - 1)/self.tau_x - self.u*self.x*self._curr_state
        self.u = self.u + dt*du
        self.x = self.x + dt*dx

    def state(self):
        return self._curr_state

    def dynamics_noiseless(self, drive, dt):
        g = self.tf(np.dot(self.w, self._curr_state) + self.bias + drive)
        if self.stp:
            g = self.u*self.x*self.tau*g
            self._step_stp_dynamics(dt)
        dr = -self._curr_state + g
        return dr/self.tau

    def dynamics_poisson(self, drive, dt):
        g = self.tf(np.dot(self.w, self._curr_state) + self.bias + drive)
        if self.stp:
            g = self.u*self.x*self.tau*g
            self._step_stp_dynamics(dt)
        spks = np.random.poisson(g*dt)
        dr = -self._curr_state + spks
        return dr/self.tau

    def dynamics_poisson_f(self, drive, dt):
        self._curr_spks = np.random.poisson(self.tf(self._curr_state)*dt)
        g = np.dot(self.w, self._curr_spks) + self.bias + drive
        if self.stp:
            g = self.u*self.x*self.tau*g
            self._step_stp_dynamics(dt)
        dr = -self._curr_state + g
        return dr/self.tau

    def dynamics_noiseless_f(self, drive, dt):
        self._curr_spks = self.tf(self._curr_state)
        g = np.dot(self.w, self._curr_spks) + self.bias + drive
        if self.stp:
            g = self.u*self.x*self.tau*g
            self._step_stp_dynamics(dt)
        dr = -self._curr_state + g
        return dr/self.tau

    def get_state(self):
        return self._curr_state

    def get_spks(self):
        return self._curr_spks

    def get_output(self, dynamics_f=True):
        if dynamics_f:
            out = self.get_spks()
        else:
            out = self.get_state()
        return out

    def get_time(self):
        return self._curr_time

    def initialize_network(self, init_scale=.01):
        self._curr_state = np.random.rand(self.n_neurs)*init_scale
        self._curr_spks = np.zeros(self.n_neurs)
        self._curr_time = 0
        if self.stp:
            self.x = np.ones(self.n_neurs)
            self.u = np.ones(self.n_neurs)*self.big_u
    
    def begin_integration(self, t_end, dt, init_cond=None, init_scale=.01,
                          dynamics=dynamics_noiseless, drive_func=None):
        self.initialize_network()
        out = self.integrate_until(t_end, dt, dynamics=dynamics,
                                   drive_func=drive_func)
        return out

    def iterate_step(self, drive, dt, dynamics):
        self._curr_state = (self._curr_state
                            + dt*dynamics(drive, dt))
        self._curr_time = self._curr_time + dt

    def focus(self):
        return self.thetas[np.argmax(self._curr_state)]
        
    def integrate_until(self, t_end, dt, keep=10, dynamics='noiseless',
                        drive_func=None):
        if dynamics == 'noiseless':
            dynamics = self.dynamics_noiseless_f
        elif dynamics == 'poisson':
            dynamics = self.dynamics_poisson_f
        if drive_func is None:
            drive_func = lambda t, curr, dt: 0
        j = 0
        out = get_steps_and_containers(self._curr_time, t_end, dt, keep,
                                       self.n_neurs)
        steps_to_end, steps_to_keep, interval = out[:3]
        trace, focus, store_drive, time = out[3:]
        for i in range(steps_to_end):
            drive = drive_func(self._curr_time, self._curr_state, dt)
            self.iterate_step(drive, dt, dynamics)
            if (i % interval) == 0 and j < steps_to_keep:
                trace[j] = self.get_output()
                focus[j] = self.focus()
                store_drive[j] = drive
                time[j] = self._curr_time
                j = j + 1
        return trace, focus, store_drive, time


# Seung functions

def _f0(x):
    return (np.sin(x) - x*np.cos(x))/np.pi

def _f1(x):
    return (x - .5*np.sin(2*x))/(2*np.pi)

def _f1_asym(x):
    return (x**3)/(3*np.pi)

def get_zero(v, z, eps=.001):
    cands = v[z < eps]
    if len(cands) > 0:
        tc = cands[0]
    else:
        tc = np.nan
    return tc

def theta_c(j1, k1, phi, eps=.001, n_tcs=1000000):
    tc = np.linspace(0, 100, n_tcs)
    f1 = _f1(tc)
    z = np.abs(f1*(j1*np.cos(phi) + np.sqrt(k1**2 - (j1**2)*np.sin(phi)**2))
               - 1)
    tc = get_zero(tc, z, eps=eps)
    return z, tc

def theta_c_r(j1, phi, n_tcrs=1000000, eps=.001):
    tcrs = np.linspace(0, 100, n_tcrs)
    f1 = _f1(tcrs)
    z = np.abs(j1*f1*np.cos(phi) - 1)
    tcr = get_zero(tcrs, z, eps=eps)
    return z, tcr

def delta_bc(j0, j1, k0, k1, tcr):
    f0 = _f0(tcr)
    num = (k0 - j0)*f0 + k1/j1 - np.cos(tcr)
    denom = (j0 + k0)*f0 + k1/j1 + np.cos(tcr)
    return num/denom

def vsat_dbc(j0, j1, k0, k1, phi, tau=10):
    tcr = theta_c_r(j1, phi)[1]
    dbc = delta_bc(j0, j1, k0, k1, tcr)
    vsat = np.tan(phi)/tau
    return vsat/dbc

def vmin_db(j0, j1, k0, k1, phi, tau=10):
    tc = theta_c(j1, k1, phi)[1]
    t1 = 2*j1*np.sin(phi)*np.sin(tc)/tau
    t2 = -(j0 + k0)*_f0(tc) - np.cos(tc)
    denom = np.pi - tc*(j0 - k0)
    return t1*t2/denom

def plot_slopes(j0, j1, k0, k1, phis):
    vsbc = np.zeros_like(phis)
    vb = np.zeros_like(phis)
    for i, phi in enumerate(phis):
        vsbc[i] = vsat_dbc(j0, j1, k0, k1, phi)
        vb[i] = vmin_db(j0, j1, k0, j1, phi)
    f = plt.figure()
    ax = f.add_subplot(1,1,1)
    ax.plot(phis, vsbc, 'o', label='sat slope')
    ax.plot(phis, vb, 'o', label='orig slope')
    ax.legend()
    return vsbc, vb

def plot_theta_c(j1, k1, phis, n_tcs=10000):
    f = plt.figure()
    ax = f.add_subplot(1,1,1)
    dcs = np.zeros_like(phis)
    for i, p in enumerate(phis):
        dcs[i] = theta_c(j1, k1, p, n_tcs=n_tcs)[1]
    ax.plot(phis, dcs, 'o')
    ax.legend(frameon=False)
    return ax

