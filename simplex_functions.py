import numpy as np
import matplotlib.pyplot as plt
import sgp4.api
from sgp4.api import Satrec, jday, days2mdhms
import datetime as dt
from scipy.optimize import minimize, curve_fit
from joblib import Parallel,delayed
import glob

def extract_numbers(line1, line2):
    epoch = float(line1[18:32])
    drag = float(line1[53:59])*10**(float(line1[59:61]))
    inclination = float(line2[8:16])
    RAAN = float(line2[17:25])
    eccentricity = float('.'+line2[26:33])
    AOP = float(line2[34:42])
    mean_anomaly = float(line2[43:51])
    mean_motion = float(line2[52:63])
    return np.array([epoch,drag,inclination,RAAN,eccentricity,AOP,mean_anomaly,mean_motion])

def drag_to_string(num):
    sign = " " if num >= 0 else "-"
    num = abs(num)
    exponent = 0
    while num >= 100000: 
        num /= 10
        exponent += 1
    while num < 10000 and num != 0: 
        num *= 10
        exponent -= 1
    num = round(num)
    return f"{sign}{int(num):05d}{exponent:+d}"

def eccentricity_to_string(num):
    num_str = f"{num:.7f}".replace("0.","")
    return num_str[:7]

def state_vector_to_TLE(state_vector,reference):
    new_text_line_1 = reference[0][0:53]+drag_to_string(state_vector[0])+reference[0][61:]
    new_text_line_2 = reference[1][0:8]+f"{state_vector[1]:8.4f}"+" "+f"{state_vector[2]:8.4f}"+" "+eccentricity_to_string(state_vector[3])+" "+f"{state_vector[4]:8.4f}"+" "+f"{state_vector[5]:8.4f}"+" "+f"{state_vector[6]:11.8f}"+reference[1][63:]
    return [new_text_line_1,new_text_line_2]

def TLEtime_to_julian(n):
    year_no = int(np.floor(n/1000))
    if year_no<57:
        year = 2000+year_no
    else:
        year = 1900 + year_no

    full_day = n - year_no*1000
    mon,day,hr,minute,sec = days2mdhms(year_no,full_day)
    jd,fr = jday(year,mon,day,hr,minute,sec)
    return (jd,fr)

def tle_to_array(tlefiles_folder,sat_name): 
    txt_files = glob.glob(tlefiles_folder+"/*.txt")
    lines=[]
    for file_path in txt_files:
        with open(file_path,"r") as file:
            sat_lines = file.readlines()
            for i, line in enumerate(sat_lines): 
                if line.strip()==sat_name and i+2<len(sat_lines):
                    line1 = sat_lines[i+1].strip()
                    line2 = sat_lines[i+2].strip()
                    lines.extend([line1,line2])
                    break

    # Checking whether the tlefile format is fine
    if len(lines)%2!=0:
        return print("Error in TLE file")
    
    tle_data=[]
    times=[]
    for i in range(0,len(lines),2):
        single_tle = [lines[i].strip(),lines[i+1].strip()]
        times = extract_numbers(lines[i],lines[i+1])[0]
        tle_data.append([single_tle,times])
    
    tle_data.sort(key=lambda x:x[1])
    data=[]
    times=[]
    for i in range(len(tle_data)):
        data.append(tle_data[i][0])
        times.append(TLEtime_to_julian(tle_data[i][1]))
    return (data,np.array(times))

def all_pos_vel(data,times):
    wrappers = [Satrec.twoline2rv(tle[0],tle[1]) for tle in data]
    all_TLE=[]
    for i in range(len(times)):
        error, position, velocity = wrappers[i].sgp4(times[i][0],times[i][1])
        if error==0:
            new=np.array([np.array(position),np.array(velocity)])
            all_TLE.append(new)
    return np.array(all_TLE)

def ref_TLE(datatimes,num_of_TLE,fraction):
    usable_time = datatimes[1][:num_of_TLE]
    times = np.sum(usable_time,axis=1)
    if fraction == 0:
        initial_state = extract_numbers(datatimes[0][0][0],datatimes[0][0][1])[1:]
        return (initial_state,datatimes[0][0])
    if fraction == 1:
        initial_state = extract_numbers(datatimes[0][num_of_TLE-1][0],datatimes[0][num_of_TLE-1][1])[1:]
        return (initial_state,datatimes[0][num_of_TLE-1])
    else:
        ref_time = times[0]+fraction*(times[-1]-times[0])
        closest_index = min(range(len(times)), key=lambda i: abs(times[i] - ref_time))
        ref = datatimes[0][closest_index]
        initial_state = extract_numbers(ref[0],ref[1])[1:]
        return (initial_state, ref)

def loss(state_vector,ref_TLE,all_values,times):
    new_ref = state_vector_to_TLE(state_vector,ref_TLE)
    if len(new_ref[0])!=69 or len(new_ref[1])!=69:
        return np.inf
    
    all_references=[]
    error_free=[]
    ref_wrap = Satrec.twoline2rv(new_ref[0],new_ref[1])
    for i in range(len(times)):
        err,pos,vel = ref_wrap.sgp4(times[i][0],times[i][1])
        if err ==0:
            all_references.append(np.array([np.array(pos),np.array(vel)]))
            error_free.append(i)
    all_references = np.array(all_references)

    all_TLE = all_values[error_free]

    if len(all_TLE)!=len(all_references) or len(all_TLE)==0 or len(all_references)==0:
        return np.inf
    
    return np.sum((all_TLE-all_references)**2)

def loss_weighted(state_vector,ref_TLE,all_values,times):
    new_ref = state_vector_to_TLE(state_vector,ref_TLE)
    if len(new_ref[0])!=69 or len(new_ref[1])!=69:
        return np.inf
    
    all_references=[]
    error_free=[]
    ref_wrap = Satrec.twoline2rv(new_ref[0],new_ref[1])
    for i in range(len(times)):
        err,pos,vel = ref_wrap.sgp4(times[i][0],times[i][1])
        if err ==0:
            all_references.append(np.array([np.array(pos),np.array(vel)]))
            error_free.append(i)
    all_references = np.array(all_references)

    all_TLE = all_values[error_free]

    if len(all_TLE)!=len(all_references) or len(all_TLE)==0 or len(all_references)==0:
        return np.inf
    pos_weight = 1e-6
    pos_error = pos_weight*np.sum((all_TLE[:,0]-all_references[:,0])**2)
    vel_error = np.sum((all_TLE[:,1]-all_references[:,1])**2)
    return pos_error+vel_error

def loss_pos(state_vector,ref_TLE,all_values,times):
    new_ref = state_vector_to_TLE(state_vector,ref_TLE)
    if len(new_ref[0])!=69 or len(new_ref[1])!=69:
        return np.inf
    
    all_references=[]
    error_free=[]
    ref_wrap = Satrec.twoline2rv(new_ref[0],new_ref[1])
    for i in range(len(times)):
        err,pos,vel = ref_wrap.sgp4(times[i][0],times[i][1])
        if err ==0:
            all_references.append(np.array([np.array(pos),np.array(vel)]))
            error_free.append(i)
    all_references = np.array(all_references)

    all_TLE = all_values[error_free]

    if len(all_TLE)!=len(all_references) or len(all_TLE)==0 or len(all_references)==0:
        return np.inf
    return np.sum((all_TLE[:,0]-all_references[:,0])**2)

def loss_vel(state_vector,ref_TLE,all_values,times):
    new_ref = state_vector_to_TLE(state_vector,ref_TLE)
    if len(new_ref[0])!=69 or len(new_ref[1])!=69:
        return np.inf
    
    all_references=[]
    error_free=[]
    ref_wrap = Satrec.twoline2rv(new_ref[0],new_ref[1])
    for i in range(len(times)):
        err,pos,vel = ref_wrap.sgp4(times[i][0],times[i][1])
        if err ==0:
            all_references.append(np.array([np.array(pos),np.array(vel)]))
            error_free.append(i)
    all_references = np.array(all_references)

    all_TLE = all_values[error_free]

    if len(all_TLE)!=len(all_references) or len(all_TLE)==0 or len(all_references)==0:
        return np.inf
    return np.sum((all_TLE[:,1]-all_references[:,1])**2)


def all_states(initial_state,num_samples):
    scale_factors = 0.01 * initial_state
    states = np.zeros((num_samples, 7))

    for i in range(num_samples):
        states[i, :] = initial_state + np.random.normal(0, scale=scale_factors, size=7)
    return states

def optimise_state(state,ref, all_values, datatimes, initial_error):
        bounds = np.column_stack((state*0.99,state*1.01))
        result = minimize(loss,state,args=(ref,all_values,datatimes[1]),method='nelder-mead',tol=1e-2,bounds=bounds)
        if result.success and result.fun<initial_error:
            return result.x
        else: return None

def optimise_state_weighted(state,ref, all_values, datatimes, initial_error):
        bounds = np.column_stack((state*0.99,state*1.01))
        result = minimize(loss_weighted,state,args=(ref,all_values,datatimes[1]),method='nelder-mead',tol=1e-2,bounds=bounds)
        if result.success and result.fun<initial_error:
            return result.x
        else: return None

def optimise_state_pos(state,ref, all_values, datatimes, initial_error):
        bounds = np.column_stack((state*0.99,state*1.01))
        result = minimize(loss_pos,state,args=(ref,all_values,datatimes[1]),method='nelder-mead',tol=1e-2,bounds=bounds)
        if result.success and result.fun<initial_error:
            return result.x
        else: return None

def optimise_state_vel(state,ref, all_values, datatimes, initial_error):
        bounds = np.column_stack((state*0.99,state*1.01))
        result = minimize(loss_vel,state,args=(ref,all_values,datatimes[1]),method='nelder-mead',tol=1e-2,bounds=bounds)
        if result.success and result.fun<initial_error:
            return result.x
        else: return None

def monte_carlo_simplex_joblib(datatimes,initial_state,ref,all_values,num_samples):
    states = all_states(initial_state,num_samples)
    initial_error = loss(initial_state,ref,all_values,datatimes[1])
    solutions=[]
    
    solutions = Parallel(n_jobs=-1, backend="loky")(
        delayed(optimise_state)(s, ref, all_values, datatimes, initial_error) for s in states
    )
    
    return np.array([sol for sol in solutions if sol is not None])

def monte_carlo_simplex_weighted(datatimes,initial_state,ref,all_values,num_samples):
    states = all_states(initial_state,num_samples)
    initial_error = loss(initial_state,ref,all_values,datatimes[1])
    solutions=[]
    
    solutions = Parallel(n_jobs=-1, backend="loky")(
        delayed(optimise_state_weighted)(s, ref, all_values, datatimes, initial_error) for s in states
    )
    
    return np.array([sol for sol in solutions if sol is not None])

def monte_carlo_simplex_pos(datatimes,initial_state,ref,all_values,num_samples):
    states = all_states(initial_state,num_samples)
    initial_error = loss(initial_state,ref,all_values,datatimes[1])
    solutions=[]
    
    solutions = Parallel(n_jobs=-1, backend="loky")(
        delayed(optimise_state_pos)(s, ref, all_values, datatimes, initial_error) for s in states
    )
    
    return np.array([sol for sol in solutions if sol is not None])

def monte_carlo_simplex_vel(datatimes,initial_state,ref,all_values,num_samples):
    states = all_states(initial_state,num_samples)
    initial_error = loss(initial_state,ref,all_values,datatimes[1])
    solutions=[]
    
    solutions = Parallel(n_jobs=-1, backend="loky")(
        delayed(optimise_state_vel)(s, ref, all_values, datatimes, initial_error) for s in states
    )
    
    return np.array([sol for sol in solutions if sol is not None])

def normal(x,mu,sigma,A):
    return A*np.exp(-(x-mu)**2/(2*sigma**2))/(np.sqrt(2*np.pi*sigma**2))

def fit_normal(data,bins=50):
    hist,bin_edges = np.histogram(data,bins=bins,density=True)
    bin_centers = (bin_edges[:-1]+bin_edges[1:])/2

    try:
        initial_guess = [np.mean(data),np.std(data),np.max(hist)]
        popt,_ = curve_fit(normal,bin_centers,hist,p0=initial_guess)
        return popt
    except (ValueError,RuntimeError):
        max_bin_index = np.argmax(hist)
        return [bin_centers[max_bin_index],np.std(data),np.max(hist)]

def optimised_TLE(MCS_solutions,ref_TLE):
    optimised_state=[]
    for i in range(7):
        optimised_state.append(fit_normal(MCS_solutions[:,i])[0])
    return state_vector_to_TLE(optimised_state,ref_TLE)

def propogated_loss(opt_TLE,ref_TLE,datatimes):
    # all julian day times
    data = datatimes[0]
    time = datatimes[1]

    #optimised values
    all_optimised=[]
    optimised_wrapper = Satrec.twoline2rv(opt_TLE[0],opt_TLE[1])
    for i in range(len(time)):
        opt_error, opt_position, opt_velocity = optimised_wrapper.sgp4(time[i][0],time[i][1])
        if opt_error==0:
            opt_vals=np.array([np.array(opt_position),np.array(opt_velocity)])
            all_optimised.append(opt_vals)
    all_optimised = np.array(all_optimised)

    #propagated values from the reference TLE
    all_reference=[]
    reference_wrapper = Satrec.twoline2rv(ref_TLE[0],ref_TLE[1])
    for i in range(len(time)):
        ref_error, ref_position, ref_velocity = reference_wrapper.sgp4(time[i][0],time[i][1])
        if ref_error==0:
            new_vals=np.array([np.array(ref_position),np.array(ref_velocity)])
            all_reference.append(new_vals)
    all_reference = np.array(all_reference)

    #expected value from all TLEs
    all_TLE=[]
    tle_wrappers = [Satrec.twoline2rv(tle[0],tle[1]) for tle in data]
    for i in range(len(time)):
        error, position, velocity = tle_wrappers[i].sgp4(time[i][0],time[i][1])
        if error==0:
            new=np.array([np.array(position),np.array(velocity)])
            all_TLE.append(new)
    all_TLE=np.array(all_TLE)

    return (time,all_optimised,all_reference,all_TLE)