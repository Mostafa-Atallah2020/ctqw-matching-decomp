OPENQASM 2.0;
include "qelib1.inc";
gate mcx_vchain_dg q0,q1,q2,q3,q4,q5,q6 { h q6; t q6; cx q2,q6; tdg q6; cx q5,q6; h q5; t q5; cx q0,q5; tdg q5; cx q1,q5; t q5; cx q0,q5; tdg q5; h q5; cx q5,q6; t q6; cx q2,q6; tdg q6; h q6; ccx q3,q6,q4; h q6; t q6; cx q2,q6; tdg q6; cx q5,q6; h q5; t q5; cx q0,q5; tdg q5; cx q1,q5; t q5; cx q0,q5; tdg q5; h q5; cx q5,q6; t q6; cx q2,q6; tdg q6; h q6; ccx q3,q6,q4; }
gate mcx_vchain q0,q1,q2,q3,q4,q5,q6 { ccx q3,q6,q4; h q6; t q6; cx q2,q6; tdg q6; cx q5,q6; h q5; t q5; cx q0,q5; tdg q5; cx q1,q5; t q5; cx q0,q5; tdg q5; h q5; cx q5,q6; t q6; cx q2,q6; tdg q6; h q6; ccx q3,q6,q4; h q6; t q6; cx q2,q6; tdg q6; cx q5,q6; h q5; t q5; cx q0,q5; tdg q5; cx q1,q5; t q5; cx q0,q5; tdg q5; h q5; cx q5,q6; t q6; cx q2,q6; tdg q6; h q6; }
gate c8rx(param0) q0,q1,q2,q3,q4,q5,q6,q7,q8 { h q8; ccx q3,q5,q8; h q5; t q5; cx q2,q5; tdg q5; cx q4,q5; h q4; t q4; cx q0,q4; tdg q4; cx q1,q4; t q4; cx q0,q4; tdg q4; h q4; cx q4,q5; t q5; cx q2,q5; tdg q5; h q5; ccx q3,q5,q8; h q5; t q5; cx q2,q5; tdg q5; cx q4,q5; h q4; t q4; cx q0,q4; tdg q4; cx q1,q4; t q4; cx q0,q4; tdg q4; h q4; cx q4,q5; t q5; cx q2,q5; tdg q5; h q5; rz(-pi/16) q8; mcx_vchain_dg q4,q5,q6,q7,q8,q2,q3; rz(pi/16) q8; mcx_vchain q0,q1,q2,q3,q8,q4,q5; rz(-pi/16) q8; mcx_vchain q4,q5,q6,q7,q8,q2,q3; rz(pi/16) q8; h q8; }
gate c8rx_o253(param0) q0,q1,q2,q3,q4,q5,q6,q7,q8 { x q1; c8rx(pi/4) q0,q1,q2,q3,q4,q5,q6,q7,q8; x q1; }
gate c8rx_o87(param0) q0,q1,q2,q3,q4,q5,q6,q7,q8 { x q3; x q5; x q7; c8rx(pi/4) q0,q1,q2,q3,q4,q5,q6,q7,q8; x q3; x q5; x q7; }
qreg q[9];
c8rx_o253(pi/4) q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8];
c8rx_o87(pi/4) q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8];
