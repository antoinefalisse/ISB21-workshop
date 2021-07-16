import numpy as np

# These functions were copied from their MATLAB counterparts.

def getJointPositions(l1,l2,l3,l4,l5,q1,q2,q3,q4,q5):

    t2 = np.pi/2.0
    t3 = q1+t2
    t4 = q2+t2
    t5 = q3+t2
    t6 = q4+t2
    t7 = q5+t2
    t8 = np.cos(t3)
    t9 = np.cos(t4)
    t10 = np.cos(t6)
    t11 = np.sin(t3)
    t12 = np.sin(t4)
    t13 = np.sin(t6)
    t14 = l1*t8
    t15 = l2*t9
    t16 = l4*t10
    t17 = l1*t11
    t18 = l2*t12
    t19 = l4*t13
    t20 = -t16
    t21 = -t19
    jointPositions = [t14, t17, t14+t15, t17+t18, t14+t15+l3*np.cos(t5), t17+t18+l3*np.sin(t5), t14+t15+t20, t17+t18+t21, t14+t15+t20-l5*np.cos(t7), t17+t18+t21-l5*np.sin(t7)]

    return jointPositions

def getJointVelocities(dq1,dq2,dq3,dq4,dq5,l1,l2,l3,l4,l5,q1,q2,q3,q4,q5):

    t2 = np.pi/2.0
    t3 = q1+t2
    t4 = q2+t2
    t5 = q3+t2
    t6 = q4+t2
    t7 = q5+t2
    t8 = np.cos(t3)
    t9 = np.cos(t4)
    t10 = np.cos(t6)
    t11 = np.sin(t3)
    t12 = np.sin(t4)
    t13 = np.sin(t6)
    t14 = dq1*l1*t8
    t15 = dq2*l2*t9
    t16 = dq4*l4*t10
    t17 = dq1*l1*t11
    t18 = dq2*l2*t12
    t19 = dq4*l4*t13
    t20 = -t16
    t21 = -t17
    t22 = -t18
    jointVelocities = [t21, t14, t21+t22, t14+t15, t21+t22-dq3*l3*np.sin(t5), t14+t15+dq3*l3*np.cos(t5), t19+t21+t22, t14+t15+t20, t19+t21+t22+dq5*l5*np.sin(t7), t14+t15+t20-dq5*l5*np.cos(t7)] 

    return jointVelocities

def getRelativeJointAngles(q1,q2,q3,q4,q5):

    t2 = -q3
    relativeJointAngles = [q1,q1-q2,q2+t2,q4+t2,-q4+q5]
    
    return relativeJointAngles

def getRelativeJointAngularVelocities(dq1,dq2,dq3,dq4,dq5):

    t2 = -dq3
    relativeJointAngularVelocities = [dq1,dq1-dq2,dq2+t2,dq4+t2,-dq4+dq5]
    
    return relativeJointAngularVelocities
