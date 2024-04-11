import 'package:flutter/material.dart';
import 'package:adobe_xd/pinned.dart';
import './HomeFirstAID1.dart';
import 'package:adobe_xd/page_link.dart';
import 'package:flutter_svg/flutter_svg.dart';

class ModificaProdotto extends StatelessWidget {
  ModificaProdotto({
    Key? key, // Changed from Key to Key?
  }) : super(key: key);
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xffffffff),
      body: Stack(
        children: <Widget>[
          Container(
            color: const Color(0xffecf3e8),
          ),
          Pinned.fromPins(
            Pin(size: 75.0, start: 46.0),
            Pin(size: 24.0, start: 22.0),
            child: Text(
              '1° Piano',
              style: TextStyle(
                fontFamily: 'Lato',
                fontSize: 20,
                color: const Color(0xff27312f),
                fontWeight: FontWeight.w700,
              ),
              softWrap: false,
            ),
          ),
          Pinned.fromPins(
            Pin(size: 24.3, start: 14.0),
            Pin(size: 24.3, start: 21.7),
            child: PageLink(
              links: [
                PageLinkInfo(
                  pageBuilder: () => HomeFirstAID1(),
                ),
              ],
              child: Stack(
                children: <Widget>[
                  SizedBox.expand(
                      child: SvgPicture.string(
                    _svg_hjl211,
                    allowDrawingOutsideViewBox: true,
                    fit: BoxFit.fill,
                  )),
                ],
              ),
            ),
          ),
          Pinned.fromPins(
            Pin(size: 21.4, end: 53.6),
            Pin(size: 22.0, start: 22.9),
            child: Stack(
              children: <Widget>[
                Center(
                  child: Container(
                    width: 5.0,
                    height: 5.0,
                    decoration: BoxDecoration(
                      color: const Color(0xff27312f),
                      borderRadius:
                          BorderRadius.all(Radius.elliptical(9999.0, 9999.0)),
                    ),
                  ),
                ),
                SizedBox.expand(
                    child: SvgPicture.string(
                  _svg_yfxl45,
                  allowDrawingOutsideViewBox: true,
                  fit: BoxFit.fill,
                )),
              ],
            ),
          ),
          Pinned.fromPins(
            Pin(size: 24.3, end: 14.7),
            Pin(size: 24.3, start: 21.7),
            child: Stack(
              children: <Widget>[
                SizedBox.expand(
                    child: SvgPicture.string(
                  _svg_jlq,
                  allowDrawingOutsideViewBox: true,
                  fit: BoxFit.fill,
                )),
              ],
            ),
          ),
          Pinned.fromPins(
            Pin(size: 73.0, middle: 0.6667),
            Pin(size: 15.0, end: 49.0),
            child: Text(
              'Battery 100%',
              style: TextStyle(
                fontFamily: 'Lato',
                fontSize: 12,
                color: const Color(0xff27312f),
              ),
              softWrap: false,
            ),
          ),
          Pinned.fromPins(
            Pin(size: 106.0, middle: 0.7386),
            Pin(size: 15.0, end: 19.0),
            child: Text(
              'ID: 2A5FB2F2C4A3',
              style: TextStyle(
                fontFamily: 'Lato',
                fontSize: 12,
                color: const Color(0xff27312f),
              ),
              softWrap: false,
            ),
          ),
          Pinned.fromPins(
            Pin(size: 94.0, middle: 0.7107),
            Pin(size: 15.0, end: 34.0),
            child: Text(
              'Last Report 11:20',
              style: TextStyle(
                fontFamily: 'Lato',
                fontSize: 12,
                color: const Color(0xff27312f),
              ),
              softWrap: false,
            ),
          ),
          Pinned.fromPins(
            Pin(size: 120.0, middle: 0.274),
            Pin(size: 40.0, end: 33.0),
            child: Container(
              decoration: BoxDecoration(
                image: DecorationImage(
                  image: const AssetImage(''),
                  fit: BoxFit.fill,
                ),
              ),
            ),
          ),
          Pinned.fromPins(
            Pin(size: 66.0, middle: 0.3353),
            Pin(size: 15.0, end: 22.0),
            child: Text(
              'First AID Kit',
              style: TextStyle(
                fontFamily: 'Lato',
                fontSize: 12,
                color: const Color(0xff27312f),
              ),
              softWrap: false,
            ),
          ),
          Pinned.fromPins(
            Pin(size: 143.0, middle: 0.5019),
            Pin(size: 143.0, start: 100.0),
            child: Stack(
              children: <Widget>[
                Container(
                  decoration: BoxDecoration(
                    color: const Color(0xffffffff),
                    borderRadius:
                        BorderRadius.all(Radius.elliptical(9999.0, 9999.0)),
                    border:
                        Border.all(width: 1.0, color: const Color(0xff707070)),
                  ),
                ),
              ],
            ),
          ),
          Pinned.fromPins(
            Pin(start: 27.0, end: 27.0),
            Pin(size: 56.0, middle: 0.4043),
            child: Container(
              decoration: BoxDecoration(
                color: const Color(0xff27312f),
                borderRadius: BorderRadius.circular(10.0),
              ),
            ),
          ),
          Pinned.fromPins(
            Pin(size: 59.0, start: 27.0),
            Pin(size: 17.0, middle: 0.3577),
            child: Text(
              'Scadenza',
              style: TextStyle(
                fontFamily: 'Lato',
                fontSize: 14,
                color: const Color(0xff27312f),
                fontWeight: FontWeight.w700,
              ),
              softWrap: false,
            ),
          ),
          Pinned.fromPins(
            Pin(start: 27.0, end: 27.0),
            Pin(size: 56.0, middle: 0.6232),
            child: Container(
              decoration: BoxDecoration(
                color: const Color(0xff27312f),
                borderRadius: BorderRadius.circular(10.0),
              ),
            ),
          ),
          Pinned.fromPins(
            Pin(start: 40.0, end: 39.0),
            Pin(size: 34.0, middle: 0.62),
            child: Text(
              'Farmaco a base di Ibuprufene analgesico per mal di testa, doloriarticolari, mal di denti e altro.',
              style: TextStyle(
                fontFamily: 'Lato',
                fontSize: 14,
                color: const Color(0xffecf3e8),
                fontWeight: FontWeight.w700,
              ),
            ),
          ),
          Pinned.fromPins(
            Pin(start: 27.0, end: 27.0),
            Pin(size: 56.0, middle: 0.7344),
            child: Container(
              decoration: BoxDecoration(
                color: const Color(0xff27312f),
                borderRadius: BorderRadius.circular(10.0),
              ),
            ),
          ),
          Pinned.fromPins(
            Pin(start: 40.0, end: -48.0),
            Pin(size: 17.0, middle: 0.7246),
            child: Text(
              'Conservare in un luogo privo di umidità e non vicino a fonti di calore',
              style: TextStyle(
                fontFamily: 'Lato',
                fontSize: 14,
                color: const Color(0xffecf3e8),
                fontWeight: FontWeight.w700,
              ),
              softWrap: false,
            ),
          ),
          Pinned.fromPins(
            Pin(size: 53.0, start: 27.0),
            Pin(size: 56.0, end: 130.0),
            child: Container(
              decoration: BoxDecoration(
                color: const Color(0xff27312f),
                borderRadius: BorderRadius.circular(10.0),
              ),
            ),
          ),
          Pinned.fromPins(
            Pin(size: 22.0, start: 43.0),
            Pin(size: 17.0, end: 149.0),
            child: Text(
              '22°',
              style: TextStyle(
                fontFamily: 'Lato',
                fontSize: 14,
                color: const Color(0xffecf3e8),
                fontWeight: FontWeight.w700,
              ),
              softWrap: false,
            ),
          ),
          Pinned.fromPins(
            Pin(size: 74.0, middle: 0.6391),
            Pin(size: 56.0, end: 84.0),
            child: Container(
              decoration: BoxDecoration(
                color: const Color(0xff27312f),
                borderRadius: BorderRadius.circular(10.0),
              ),
            ),
          ),
          Pinned.fromPins(
            Pin(size: 33.0, middle: 0.6253),
            Pin(size: 17.0, end: 103.0),
            child: Text(
              'Salva',
              style: TextStyle(
                fontFamily: 'Lato',
                fontSize: 14,
                color: const Color(0xffecf3e8),
                fontWeight: FontWeight.w700,
              ),
              softWrap: false,
            ),
          ),
          Pinned.fromPins(
            Pin(size: 143.0, middle: 0.5019),
            Pin(size: 143.0, start: 100.0),
            child: Stack(
              children: <Widget>[
                Container(
                  decoration: BoxDecoration(
                    image: DecorationImage(
                      image: const AssetImage(''),
                      fit: BoxFit.fill,
                    ),
                  ),
                  margin: EdgeInsets.fromLTRB(-74.0, -11.0, -77.0, 0.0),
                ),
                Container(
                  decoration: BoxDecoration(
                    color: const Color(0xffffffff),
                    borderRadius:
                        BorderRadius.all(Radius.elliptical(9999.0, 9999.0)),
                    border:
                        Border.all(width: 1.0, color: const Color(0xff707070)),
                  ),
                ),
              ],
            ),
          ),
          Align(
            alignment: Alignment(0.0, -0.385),
            child: SizedBox(
              width: 116.0,
              height: 24.0,
              child: Text(
                'Nurofenteen',
                style: TextStyle(
                  fontFamily: 'Lato',
                  fontSize: 20,
                  color: const Color(0xff27312f),
                  fontWeight: FontWeight.w700,
                ),
                softWrap: false,
              ),
            ),
          ),
          Pinned.fromPins(
            Pin(size: 76.0, start: 42.0),
            Pin(size: 17.0, middle: 0.4091),
            child: Text(
              '12/08/2023',
              style: TextStyle(
                fontFamily: 'Lato',
                fontSize: 14,
                color: const Color(0xffecf3e8),
                fontWeight: FontWeight.w700,
              ),
              softWrap: false,
            ),
          ),
          Pinned.fromPins(
            Pin(size: 74.0, start: 27.0),
            Pin(size: 17.0, middle: 0.5669),
            child: Text(
              'Descrizione',
              style: TextStyle(
                fontFamily: 'Lato',
                fontSize: 14,
                color: const Color(0xff27312f),
                fontWeight: FontWeight.w700,
              ),
              softWrap: false,
            ),
          ),
          Pinned.fromPins(
            Pin(size: 127.0, start: 27.0),
            Pin(size: 17.0, middle: 0.6731),
            child: Text(
              'Note Conservazione',
              style: TextStyle(
                fontFamily: 'Lato',
                fontSize: 14,
                color: const Color(0xff27312f),
                fontWeight: FontWeight.w700,
              ),
              softWrap: false,
            ),
          ),
          Pinned.fromPins(
            Pin(size: 177.0, start: 27.0),
            Pin(size: 17.0, middle: 0.7783),
            child: Text(
              'Temperatura Conservazione',
              style: TextStyle(
                fontFamily: 'Lato',
                fontSize: 14,
                color: const Color(0xff27312f),
                fontWeight: FontWeight.w700,
              ),
              softWrap: false,
            ),
          ),
          Pinned.fromPins(
            Pin(size: 74.0, middle: 0.3609),
            Pin(size: 56.0, end: 84.0),
            child: Container(
              decoration: BoxDecoration(
                color: const Color(0xff27312f),
                borderRadius: BorderRadius.circular(10.0),
                border: Border.all(width: 3.0, color: const Color(0xfff84545)),
              ),
            ),
          ),
          Pinned.fromPins(
            Pin(size: 46.0, middle: 0.3716),
            Pin(size: 17.0, end: 103.0),
            child: Text(
              'Elimina',
              style: TextStyle(
                fontFamily: 'Lato',
                fontSize: 14,
                color: const Color(0xffecf3e8),
                fontWeight: FontWeight.w700,
              ),
              softWrap: false,
            ),
          ),
          Pinned.fromPins(
            Pin(start: 27.0, end: 27.0),
            Pin(size: 56.0, middle: 0.5156),
            child: Container(
              decoration: BoxDecoration(
                color: const Color(0xff27312f),
                borderRadius: BorderRadius.circular(10.0),
              ),
            ),
          ),
          Pinned.fromPins(
            Pin(size: 56.0, start: 27.0),
            Pin(size: 17.0, middle: 0.464),
            child: Text(
              'Quantità',
              style: TextStyle(
                fontFamily: 'Lato',
                fontSize: 14,
                color: const Color(0xff27312f),
                fontWeight: FontWeight.w700,
              ),
              softWrap: false,
            ),
          ),
          Pinned.fromPins(
            Pin(size: 16.0, start: 42.0),
            Pin(size: 17.0, middle: 0.5154),
            child: Text(
              '12',
              style: TextStyle(
                fontFamily: 'Lato',
                fontSize: 14,
                color: const Color(0xffecf3e8),
                fontWeight: FontWeight.w700,
              ),
              softWrap: false,
            ),
          ),
        ],
      ),
    );
  }
}

const String _svg_hjl211 =
    '<svg viewBox="1.0 1.0 24.3 24.3" ><path  d="M 13.16260623931885 25.32521057128906 C 6.417888164520264 25.32521057128906 1 19.90732383728027 1 13.16260623931885 C 1 6.417887687683105 6.417888164520264 1 13.16260623931885 1 C 15.48455905914307 1 17.80651092529297 1.663414835929871 19.79675483703613 2.990244626998901 C 20.34959983825684 3.321951866149902 20.46016883850098 3.985366821289062 20.12845993041992 4.538212299346924 C 19.79675483703613 5.091058254241943 19.13333892822266 5.201627254486084 18.58049201965332 4.869920253753662 C 16.92195510864258 3.764228582382202 15.04228115081787 3.211382865905762 13.16260623931885 3.211382865905762 C 7.634148597717285 3.211382865905762 3.211382865905762 7.634148597717285 3.211382865905762 13.16260623931885 C 3.211382865905762 18.69106292724609 7.634148597717285 23.11382865905762 13.16260623931885 23.11382865905762 C 18.69106292724609 23.11382865905762 23.11382865905762 18.69106292724609 23.11382865905762 13.16260623931885 C 23.11382865905762 11.17236137390137 22.56098175048828 9.182116508483887 21.3447208404541 7.523579597473145 C 21.01301574707031 6.970733642578125 21.12358283996582 6.307319164276123 21.67642784118652 5.975611209869385 C 22.22927284240723 5.643903732299805 22.8926887512207 5.754473209381104 23.22439575195312 6.307319164276123 C 24.55122566223145 8.297564506530762 25.32520866394043 10.7300853729248 25.32520866394043 13.16260623931885 C 25.32521057128906 19.90732383728027 19.90732383728027 25.32521057128906 13.16260623931885 25.32521057128906 Z M 18.69106292724609 9.845532417297363 L 16.47967910766602 9.845532417297363 L 16.47967910766602 7.634148597717285 C 16.47967910766602 6.970733642578125 16.03740310668945 6.528457164764404 15.37398910522461 6.528457164764404 L 10.95122337341309 6.528457164764404 C 10.28780841827393 6.528457164764404 9.845532417297363 6.970733642578125 9.845532417297363 7.634148597717285 L 9.845532417297363 9.845532417297363 L 7.634148597717285 9.845532417297363 C 6.970733642578125 9.845532417297363 6.528457164764404 10.28780841827393 6.528457164764404 10.95122337341309 L 6.528457164764404 15.37398910522461 C 6.528457164764404 16.03740310668945 6.970733642578125 16.47967910766602 7.634148597717285 16.47967910766602 L 9.845532417297363 16.47967910766602 L 9.845532417297363 18.69106292724609 C 9.845532417297363 19.35447692871094 10.28780841827393 19.79675483703613 10.95122337341309 19.79675483703613 L 15.37398910522461 19.79675483703613 C 16.03740310668945 19.79675483703613 16.47967910766602 19.35447692871094 16.47967910766602 18.69106292724609 L 16.47967910766602 16.47967910766602 L 18.69106292724609 16.47967910766602 C 19.35447692871094 16.47967910766602 19.79675483703613 16.03740310668945 19.79675483703613 15.37398910522461 L 19.79675483703613 10.95122337341309 C 19.79675483703613 10.28780841827393 19.35447692871094 9.845532417297363 18.69106292724609 9.845532417297363 Z" fill="#27312f" stroke="none" stroke-width="1" stroke-miterlimit="4" stroke-linecap="butt" /></svg>';
const String _svg_yfxl45 =
    '<svg viewBox="3.0 2.4 21.4 22.0" ><path  d="M 23.9393253326416 15.52492904663086 L 23.9168758392334 15.5067777633667 L 22.40943336486816 14.32461071014404 C 22.21578407287598 14.17141723632812 22.1074104309082 13.93459892272949 22.11807060241699 13.68791198730469 L 22.11807060241699 13.13575744628906 C 22.10836601257324 12.89063358306885 22.21692276000977 12.65575408935547 22.40991020202637 12.50431251525879 L 23.9168758392334 11.3216667175293 L 23.9393253326416 11.30351543426514 C 24.41935348510742 10.90363121032715 24.5386905670166 10.2147102355957 24.22113418579102 9.676664352416992 L 22.18111991882324 6.146877765655518 C 22.17877006530762 6.143547058105469 22.17669486999512 6.140033721923828 22.17491149902344 6.136370182037354 C 21.85595893859863 5.606298923492432 21.20416450500488 5.379325866699219 20.62495994567871 5.596632957458496 L 20.60823822021484 5.602842330932617 L 18.836181640625 6.315963268280029 C 18.60928535461426 6.40770435333252 18.35211944580078 6.384777545928955 18.14503288269043 6.254347801208496 C 17.98836517333984 6.155634880065918 17.82915115356445 6.062335014343262 17.66738891601562 5.974448680877686 C 17.45468902587891 5.859107971191406 17.30947685241699 5.649534702301025 17.27619743347168 5.409874439239502 L 17.00919532775879 3.518884181976318 L 17.00346565246582 3.484493732452393 C 16.88365745544434 2.880797624588013 16.35696792602539 2.443815469741821 15.741530418396 2.4375 L 11.65672492980957 2.4375 C 11.0316333770752 2.439500331878662 10.49854469299316 2.890762567520142 10.39335918426514 3.506943225860596 L 10.38906002044678 3.533691644668579 L 10.12301254272461 5.428502082824707 C 10.08992099761963 5.667430877685547 9.94597339630127 5.876713275909424 9.734688758850098 5.993076324462891 C 9.572457313537598 6.080453395843506 9.413156509399414 6.173168182373047 9.257045745849609 6.271062850952148 C 9.050335884094238 6.400691032409668 8.793978691101074 6.423254013061523 8.567804336547852 6.331725597381592 L 6.794315338134766 5.615260601043701 L 6.777597904205322 5.608573913574219 C 6.197518825531006 5.391016006469727 5.544827461242676 5.618895530700684 5.226212024688721 6.150221347808838 L 5.2200026512146 6.16072940826416 L 3.177122354507446 9.692901611328125 C 2.859086513519287 10.23149967193604 2.978425264358521 10.92123508453369 3.458932399749756 11.3216667175293 L 3.481381416320801 11.33981704711914 L 4.988823413848877 12.52198505401611 C 5.182473659515381 12.67517852783203 5.290844917297363 12.91199684143066 5.280186176300049 13.15868282318115 L 5.280186176300049 13.71084022521973 C 5.289889335632324 13.95596218109131 5.181332588195801 14.19084167480469 4.988345623016357 14.34228515625 L 3.481380939483643 15.52492904663086 L 3.458931684494019 15.54307842254639 C 2.978901624679565 15.94296264648438 2.859563827514648 16.63188552856445 3.177121639251709 17.1699333190918 L 5.217136859893799 20.6997184753418 C 5.219484806060791 20.70304870605469 5.221561431884766 20.70656204223633 5.22334623336792 20.71022415161133 C 5.542296409606934 21.24029541015625 6.194091320037842 21.46726989746094 6.773300170898438 21.24996376037598 L 6.790016651153564 21.24375152587891 L 8.560639381408691 20.53063011169434 C 8.787535667419434 20.43889045715332 9.044700622558594 20.46181678771973 9.251791000366211 20.59224700927734 C 9.40845775604248 20.6912784576416 9.567670822143555 20.78457641601562 9.729433059692383 20.87214660644531 C 9.942132949829102 20.98748588562012 10.08734512329102 21.19705772399902 10.1206226348877 21.43671798706055 L 10.38619232177734 23.32771110534668 L 10.39192390441895 23.36210060119629 C 10.51195049285889 23.96684265136719 11.04020690917969 24.40413284301758 11.65672492980957 24.40909767150879 L 15.741530418396 24.40909576416016 C 16.36662101745605 24.40709495544434 16.89971160888672 23.95583152770996 17.00489616394043 23.33964920043945 L 17.00919723510742 23.31290245056152 L 17.27524566650391 21.41809272766113 C 17.30885696411133 21.17869758605957 17.45375633239746 20.96932029724121 17.66595649719238 20.85351753234863 C 17.82931137084961 20.76563262939453 17.98884391784668 20.6724910736084 18.14359855651855 20.57552909851074 C 18.35031127929688 20.44590187072754 18.60666847229004 20.42334175109863 18.83284187316895 20.51487159729004 L 20.6063289642334 21.22894668579102 L 20.62304878234863 21.23563194274902 C 21.20312309265137 21.4536018371582 21.8560791015625 21.22562980651855 22.17443466186523 20.6939868927002 C 22.17632484436035 20.69038200378418 22.17839622497559 20.68687438964844 22.18063926696777 20.68347930908203 L 24.22065734863281 17.15417289733887 C 24.5392894744873 16.6156177520752 24.42011070251465 15.92544746398926 23.9393253326416 15.52493190765381 Z M 17.51597785949707 13.60289096832275 C 17.41903495788574 15.66443824768066 15.70261383056641 17.2764835357666 13.63904571533203 17.24407768249512 C 11.57547855377197 17.211669921875 9.910527229309082 15.54651927947998 9.878366470336914 13.48294639587402 C 9.846205711364746 11.41937351226807 11.45845794677734 9.70314884185791 13.5200138092041 9.606450080871582 C 14.59368801116943 9.559181213378906 15.63784885406494 9.965047836303711 16.39774322509766 10.72503185272217 C 17.15763473510742 11.48501396179199 17.56337738037109 12.52922248840332 17.5159797668457 13.60289096832275 Z" fill="#27312f" stroke="none" stroke-width="0.09375" stroke-miterlimit="4" stroke-linecap="butt" /></svg>';
const String _svg_jlq =
    '<svg viewBox="3.0 3.0 24.3 24.3" ><path transform="translate(0.0, 0.0)" d="M 24.49774551391602 22.91558837890625 C 27.92592620849609 18.79092025756836 28.24472236633301 12.90659523010254 25.2822265625 8.435695648193359 C 22.31973457336426 3.964798212051392 16.77658462524414 1.964701890945435 11.64188098907471 3.513949394226074 C 6.507180213928223 5.063194274902344 2.995299816131592 9.795384407043457 2.999996662139893 15.15871238708496 C 2.999996662139893 17.99303436279297 3.999147415161133 20.73952484130859 5.822125434875488 22.91345596313477 L 5.806543350219727 22.93035697937012 C 5.867347240447998 23.00332069396973 5.936837673187256 23.06586265563965 5.999379634857178 23.1379566192627 C 6.077556610107422 23.2274284362793 6.161813259124756 23.31168556213379 6.24259614944458 23.39854621887207 C 6.485813140869141 23.66261100769043 6.735978126525879 23.91625213623047 6.998304843902588 24.15425491333008 C 7.078218936920166 24.22722053527832 7.160738468170166 24.29497528076172 7.241521835327148 24.36446762084961 C 7.51948356628418 24.60420799255371 7.805263042449951 24.83178901672363 8.101466178894043 25.04373359680176 C 8.139686584472656 25.06979560852051 8.174430847167969 25.10367012023926 8.212651252746582 25.1305980682373 L 8.212651252746582 25.12017631530762 C 12.38102245330811 28.05350875854492 17.9423828125 28.05350494384766 22.11074829101562 25.12017250061035 L 22.11074638366699 25.1305980682373 C 22.14896774291992 25.10367012023926 22.18284225463867 25.06979560852051 22.22193145751953 25.04373359680176 C 22.51726531982422 24.83091926574707 22.80391502380371 24.60420799255371 23.08187484741211 24.36446762084961 C 23.16265869140625 24.29497528076172 23.24517822265625 24.22635269165039 23.32509231567383 24.15425491333008 C 23.58741760253906 23.91538429260254 23.83758544921875 23.66261100769043 24.08080291748047 23.39854621887207 C 24.16158485412598 23.31168365478516 24.24497413635254 23.22742652893066 24.32401847839355 23.1379566192627 C 24.38569068908691 23.06586265563965 24.45605087280273 23.00332069396973 24.51685523986816 22.92948722839355 Z M 15.16082954406738 8.209668159484863 C 17.31962203979492 8.209668159484863 19.06966972351074 9.959715843200684 19.06966972351074 12.11850833892822 C 19.06966972351074 14.27730083465576 17.31962203979492 16.02734375 15.16082954406738 16.02734375 C 13.00203800201416 16.02734375 11.25199031829834 14.27729988098145 11.25199031829834 12.11850833892822 C 11.25199031829834 9.959714889526367 13.00203800201416 8.209668159484863 15.16082954406738 8.209668159484863 Z M 8.217862129211426 22.91558837890625 C 8.24921703338623 20.54228210449219 10.18142414093018 18.63445472717285 12.5549373626709 18.63323783874512 L 17.7667236328125 18.63323783874512 C 20.14023590087891 18.63445472717285 22.07244110107422 20.54228401184082 22.10379600524902 22.91558837890625 C 18.15801239013672 26.47127151489258 12.16364765167236 26.47127151489258 8.217862129211426 22.91558837890625 Z" fill="#27312f" stroke="none" stroke-width="1.5" stroke-miterlimit="4" stroke-linecap="butt" /></svg>';
