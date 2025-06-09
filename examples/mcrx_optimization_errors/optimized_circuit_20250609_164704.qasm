OPENQASM 2.0;
include "qelib1.inc";
gate mcx_vchain_dg q0,q1,q2,q3,q4 { h q3; cx q0,q3; p(pi/8) q3; cx q2,q3; p(-pi/8) q3; cx q1,q3; p(pi/8) q3; cx q2,q3; p(-pi/8) q3; cx q0,q3; p(pi/8) q3; cx q2,q3; p(-pi/8) q3; cx q1,q3; p(pi/8) q3; cx q2,q3; cx q0,q2; p(pi/8) q2; cx q1,q2; p(-pi/8) q2; cx q0,q2; p(pi/8) q2; cx q1,q2; cx q0,q1; p(pi/8) q1; cx q0,q1; p(-pi/8) q3; p(-pi/8) q2; p(-pi/8) q1; p(-pi/8) q0; h q3; }
gate mcx_vchain q0,q1,q2,q3,q4 { h q3; p(pi/8) q0; p(pi/8) q1; p(pi/8) q2; p(pi/8) q3; cx q0,q1; p(-pi/8) q1; cx q0,q1; cx q1,q2; p(-pi/8) q2; cx q0,q2; p(pi/8) q2; cx q1,q2; p(-pi/8) q2; cx q0,q2; cx q2,q3; p(-pi/8) q3; cx q1,q3; p(pi/8) q3; cx q2,q3; p(-pi/8) q3; cx q0,q3; p(pi/8) q3; cx q2,q3; p(-pi/8) q3; cx q1,q3; p(pi/8) q3; cx q2,q3; p(-pi/8) q3; cx q0,q3; h q3; }
gate c6rx(param0) q0,q1,q2,q3,q4,q5,q6 { h q6; h q6; p(pi/8) q0; p(pi/8) q1; p(pi/8) q2; p(pi/8) q6; cx q0,q1; p(-pi/8) q1; cx q0,q1; cx q1,q2; p(-pi/8) q2; cx q0,q2; p(pi/8) q2; cx q1,q2; p(-pi/8) q2; cx q0,q2; cx q2,q6; p(-pi/8) q6; cx q1,q6; p(pi/8) q6; cx q2,q6; p(-pi/8) q6; cx q0,q6; p(pi/8) q6; cx q2,q6; p(-pi/8) q6; cx q1,q6; p(pi/8) q6; cx q2,q6; p(-pi/8) q6; cx q0,q6; h q6; rz(-pi/16) q6; mcx_vchain_dg q3,q4,q5,q6,q2; rz(pi/16) q6; mcx_vchain q0,q1,q2,q6,q3; rz(-pi/16) q6; mcx_vchain q3,q4,q5,q6,q2; rz(pi/16) q6; h q6; }
qreg q[9];
cx q[1],q[3];
x q[7];
cx q[5],q[7];
c6rx(pi/4) q[0],q[2],q[3],q[4],q[6],q[7],q[8];
cx q[5],q[7];
x q[7];
cx q[1],q[3];
