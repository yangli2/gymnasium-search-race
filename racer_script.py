import base64
import zlib
import numpy as np
# yapf: disable
class Model:
    def __init__(self):
        b85_w = "cwQZr!IB(D5r!Xv3&9I;;0!kAL<C4HbF{VHvs|s|?lBi^O;=X0Pq0&&S>i-!J2P9Huy(q7#0i4c&g$-v4O{_-c>sdoA;{sj+f!Xt`RD(C*_(AC`#2^FBaK$~RJD0q7s|2Np%-l-(A5m_JE5icoUYqwNt(<vY54=^!U`mfH6u`_X^998iy@{_x;pB+I#}P_G+K5ILA;+^!H$FC-=ql`WS9rLHtj3(*8w2a|CZ1OKFpVJ+py7wdBJcVQ8)WNXqUx?zcXmOVUYGfhJ_k|aa|0@o2Ce_E`w5coE3{DB-n7K+h`1-JmaMh_L2G)k8syS)~G#Q)lB)KsR8LR5uepcoL>py4;SW9a`cNyi?#|&Y;BJng{do6$=WnR2SBEis41u_DN8X4L-*laCZX2Gxu01CqDIohWECo*a)(h{Uq5ca-Y4O?5Fte~ut6f?`EH>dP#}>GG|*XhrmP(UVO$`z<_8gC=z+Jq8)^McSIs!cZcUoGZa%sxA-RNeGZ9&u0onHx^gGIm{RYaJ<9>yW-v`za7+MkXOH4TU{;41vYVtE4kqN~|He#LIcG$%{cs1Jrtfi1^ztL2D$F(ZBrsM}$Cl5*cRFheV>er}IIx=pezM#%{IB9SlIMu0XsU8~j5_#kqu4PJK*gauykAZI|y|~FD$Wv_TsV<@%$$JK=M`FyR7C@^ex=wjIRbV}3ZO~tC!Vb?N7;Q6p5$ql>?hjE~BE=kA?^y)WT{TXzEocL`!h)l`ccV1Z+~Uoe2GxFb_O6PB#>hQSFzABmq@uTq@$R|m*rA2Ve{~Baz||HJKxbMwpuY)q8?<JBwi_&G99rMW&!W_|XE2Pg;*q?;Nps*a@5mw(NtbIE!n~|VgZBps!JpT#qmq@VaTA7*xKEOrA=RT_hGaib<5q^&AGeLCz&)coXxzG125J@UI`~EiHqF&z=)u$HGH^)QuW!qCl$K82<lR8=+&yy#YwEpo4va+6bIcyi2!rd`GbD^CeosZ}yUrV?=+?yEBBfyt0YdPg^w3xZv`{77)e>7=LC?3L-rG!kQf*Z~1lmOpH-NuHOAq0>chvuW^q?J+@tQ($F{+EQc-C^tULBw~eMEy7UuAc7I@RNktI?OsL{NFSXUgm{jIPDhirdf^1m(2tTv|l!fV0FU#HJms>v2l~%Z*MwHYZ!VjO+~awNPul18uRiUX*P}{Oy%$H~GWK*qbGF=rJ{8%ibPYJOM5z8*^Q-cjYAby>y+5_p4JnwBII<jQik~Ylo*Ur-=;0c4TZFQ)6&KQrw^|7f7Bv%};fg-Pn!{Of}ln;HW_;NlcXi&Pv7HAhSep9=kcaPP?VDP&id-_q|<Fw^Qsrua2dNF-q~oIXv93(D#bsD5%(b>A~&UQ{${YlTl;K@ngIkCJQ35b0w??|A)wlYYvS-F==R}ztD$p;)9#gN#d@_#K}rjIRAbrB-HI)6~`d9e(IQ7<+{G%z)`yo*u$X)&Q7Obaq92D*i+=kc%q(pOAk_T72*gg*yffhXSKI@Zk)BAlJk;##|vcKv}(z_6SmGN6kpVUg+OpUdsMys)bK>hflB0oaOb~x${|{GB*}w*^ah^AtG<a(+Hf5@BO98SJh3w_^5JCHyzXedf4~jttkEaorr~gC=pmwykrVBN8V%a%DLKm3Zzrb_<0_3_h&z)xjxx39+Ls*D+2(|+6CfNBo}`3fi0Hy_33i%WyY`x)tcahRr{csU87W-<->lLnx09Bt(RZEf;`|s+2na#%z*Ti$d50}n6bGCel)}DA5NlVux0UkZ?(BT9cps6APYA<R_l{rUVYlT)olM65i(`RPFBi^}S;ZMCoqMQ}E>wYjL~-X;Nq5IjSg*KR%_*L<@49n1L*es`C9q2Cj699ac<th?=mbchE~Lc4XV%KcmLC(Fu8KI$(9w2+0o&D+gC9B3`Sz3+9dMGBXhC1)av->VyckmJRNH~85h3@_cja-q@^KeS<uy6pmPU}e6x&qytaDIb>Vw$e;)xGPpDvsJbN7-&+4#3lPq2b<ah^WBz$uy49EV5$)o}PiziA}=>&dGmWqR@I4nBW&u?z;K&ORGQ8!u@y>(|Y*(o1>qrmS@GG56cc<!ApHZa@E{;jtV(S+4E)*{`1-v;OHlTMoba?vESdW4w1S?=Qa!x6@bMAvFK6;*KhCl+V1Gp}sqYndHrp_07VYslTINIONEt{bkShO})_cQp+qawh0c)i|cUnJJ@CC)ys4RDSRo%JQ)3De!Xm7kKg!6Ua|}Lnq1sCX|*36U&TL<vOjQw(cAtv9`LtQC{*<27>*ZLrM&<8ZD!fukFORw|MLSMq2+4M!Qo)tKfNIZmRRBuY#;B{8A%nqv}uui-<w6}Sb}|*7sAEL>f=vOA8c%*%VCx!tOZvC9NvHV;_y=_Uw8b=E+0l3Zo2#fp2OS2qs^CpJ$_-E+x45<VVMk8H;;7p^XYOw+&^F6yt-K1l<gwliwb|0^Zr7tO3UGXKYcy=7sh8F<opx4IOgS($J5Zr?=ScE>9_KH{2vK{*A)"
        b85_b = "cwPZK0>k|QPft71U79{DLms}D?*==@PRG3mN|3#s`%XSh?kB!h?u5L|1FXJo=LtUjj<LLVL!CYbmA<@@xc$DiT0}jtagV+2>{`94DR#Xkup&Oixmv!)vW~s79REDrTgp7S&|E$Sv*Nu?C6&I99EUn)k&-<JwNO5K9Lzje(}um$H8(yak%qolySBbixd**iN|Qcc<`q2R*owZC1<gC6P2Rm_hw?mMd@(*g|5HALwE#W?PhmYTAhA8P_XR#cKEyqh_2IoTO$EP7yHCBcN|e3d)9^f2E{eYNLUX>+jkrFSJrzDNM!`NCV7EON+U&ix0ZYE6KioY_$Q3<(qWnG&4A4D^u{u5!#C<&>YK}hn+Veb`0gFDaMYTP3Q%OD+DG|NUcUnGJ(lx&9)5N=rng>2GdqzGlEbl$Q^;SK|PPV+DHWfav3cx-BdEGsP90@)4m^(haS8Kf7)g->u`j9<<JI*^NUT8kpB`mzngtk2ml_))T4ct9R8!SHPW@|oQs1m&%aN)ekO{hJHWnVu3xwAdUT{k|gbG^O2J-9wnNkhG3ojpBZS3o|WpyfNWe+#{sIiS54%rCtS+MzzIo{~PQxUfCqSQS2D;($I~DAzlQa~nP-oYy?!Qm4HraM`?dp@BZgGPgd3cQ`&Z87n@_E_gl%@2Wi1n~gm@g~mPbcelNLAu+!6RtP>j;zqvzwS>N%5Jo;q%ptwRJ<z=2Pa!_P@$x-67(G5!4);BWI1aP"
        packed = np.frombuffer(zlib.decompress(base64.b85decode(b85_w)), dtype=np.uint8)
        w = np.empty(len(packed)*2, dtype=np.int8)
        w[0::2] = (packed >> 4)
        w[1::2] = (packed & 15)
        w = w.astype(np.float32) - 8.0
        
        self.w1 = w[0:640].reshape(64, 10) / 3.5554
        self.w2 = w[640:4736].reshape(64, 64) / 3.3079
        self.w3 = w[4736:5696].reshape(15, 64) / 5.8999
        
        biases = np.frombuffer(zlib.decompress(base64.b85decode(b85_b)), dtype=np.float32)
        self.b1 = biases[0:64]
        self.b2 = biases[64:128]
        self.b3 = biases[128:143]
        
        mv = np.frombuffer(zlib.decompress(base64.b85decode("c$@)H0I&ZBJ2*bFwh}!b?qEHY?Lj~6yAnQT+buoL6tg}5-|#+I$4fqM{V%--$ud0_+b=zG-Dp0oYe+uvkAFQFaF#tE@XkI@)4M*0H4QxVr?5PkDkLN")), dtype=np.float32)
        self.mean = mv[0:10]
        self.var = mv[10:20]
        self.eps = 1e-8
    def act(self, obs):
        obs = np.clip((obs - self.mean) / np.sqrt(self.var + self.eps), -10.0, 10.0)
        x = np.maximum(0, obs @ self.w1.T + self.b1)
        x = np.maximum(0, x @ self.w2.T + self.b2)
        return np.argmax(x @ self.w3.T + self.b3)
# yapf: enable
_model = Model()
def act(state):
  return _model.act(state)

