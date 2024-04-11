import 'package:flutter/material.dart';
import 'package:flutter_svg/flutter_svg.dart';


class Login extends StatefulWidget {
  @override
  _LoginState createState() => _LoginState();
}

class _LoginState extends State<Login> {
  TextEditingController _usernameController = TextEditingController();
  TextEditingController _passwordController = TextEditingController();
  bool _isLoading = false;

  Future<void> _login() async {
    setState(() {
      _isLoading = true; // Set loading to true when login starts
    });

    final response = await http.post(
      Uri.parse('http://5.196.23.212:8080/access'),
      body: {
        'username': _usernameController.text,
        'password': _passwordController.text,
      },
    );

    if (response.statusCode == 200) {
      Map<String, dynamic> data = json.decode(response.body);
      await storage.write(key: 'jwt', value: data['access_token']);
      Navigator.pushReplacement(
        context,
        MaterialPageRoute(builder: (context) => const DeviceListScreen()),
      );
    } else {
      // Bounce back timer (for example, 2 seconds)
      await Future.delayed(const Duration(seconds: 2));
      if (mounted) {
        // Ensure the widget is still in the tree
        setState(() {
          _isLoading = false; // Reset loading after delay
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      resizeToAvoidBottomInset: false, // Add this line
      backgroundColor: const Color(0xffffffff),
      body: Stack(
        children: <Widget>[
          Container(
            color: const Color(0xffecf3e8),
          ),
          Positioned(
            left: 0.0,
            right: 0.0,
            bottom: 27.0,
            child: Container(
              width: 120.0,
              height: 40.0,
              decoration: BoxDecoration(
                image: DecorationImage(
                  image: const AssetImage('assets/Log1.png'),
                  fit: BoxFit.fill,
                ),
              ),
            ),
          ),
          Positioned(
            top: 127.0,
            left: 61.0,
            right: 61.0,
            child: Container(
              height: 95.0,
              decoration: BoxDecoration(
                image: DecorationImage(
                  image: const AssetImage('assets/Log1.png'),
                  fit: BoxFit.fill,
                ),
              ),
            ),
          ),
          Align(
            alignment: Alignment(0.219, -0.432),
            child: SizedBox(
              width: 40.0,
              height: 40.0,
              child: SvgPicture.string(
                _svg_tjcyqk,
                allowDrawingOutsideViewBox: true,
                fit: BoxFit.fill,
              ),
            ),
          ),
          Positioned(
            left: 27.0,
            top: MediaQuery.of(context).size.height * 0.3852 - 12.5,
            child: Text(
              'Login',
              style: TextStyle(
                fontFamily: 'Lato',
                fontSize: 21,
                color: const Color(0xff27312f),
                fontWeight: FontWeight.w700,
              ),
            ),
          ),
          Align(
            alignment: Alignment(-0.219, -0.431),
            child: SizedBox(
              width: 40.0,
              height: 40.0,
              child: SvgPicture.string(
                _svg_raf7y,
                allowDrawingOutsideViewBox: true,
                fit: BoxFit.fill,
              ),
            ),
          ),
          Positioned(
            left: 27.0,
            right: 27.0,
            top: MediaQuery.of(context).size.height * 0.4533 - 28.0,
            child: Container(
              height: 56.0,
              decoration: BoxDecoration(
                color: const Color(0xff27312f),
                borderRadius: BorderRadius.circular(10.0),
              ),
            ),
          ),
          Positioned(
            left: 27.0,
            right: 27.0,
            top: MediaQuery.of(context).size.height * 0.4533 - 28.0,
            child: TextField(
              decoration: InputDecoration(
                filled: true,
                fillColor: const Color(0xff27312f),
                border: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(10.0),
                ),
                labelText: 'User',
                labelStyle: TextStyle(
                  fontFamily: 'Lato',
                  fontSize: 14,
                  color: const Color(0xffecf3e8),
                  fontWeight: FontWeight.w700,
                ),
              ),
              style: TextStyle(
                fontFamily: 'Lato',
                fontSize: 14,
                color: Colors.white,
              ),
            ),
          ),
          Positioned(
            left: 27.0,
            right: 27.0,
            top: MediaQuery.of(context).size.height * 0.549 - 28.0,
            child: TextField(
              obscureText: true,
              decoration: InputDecoration(
                filled: true,
                fillColor: const Color(0xff27312f),
                border: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(10.0),
                ),
                labelText: 'Password',
                labelStyle: TextStyle(
                  fontFamily: 'Lato',
                  fontSize: 14,
                  color: const Color(0xffecf3e8),
                  fontWeight: FontWeight.w700,
                ),
              ),
              style: TextStyle(
                fontFamily: 'Lato',
                fontSize: 14,
                color: Colors.white,
              ),
            ),
          ),
          Align(
            alignment: Alignment(0.0, 0.289),
            child: ElevatedButton(
              style: ElevatedButton.styleFrom(
                primary: const Color(0xff27312f),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(10.0),
                ),
              ),
              onPressed: _isLoading
                  ? null
                  : _login, // Disable button if _isLoading is true
              child: _isLoading
                  ? CircularProgressIndicator(
                      color: const Color(
                          0xffecf3e8)) // Show a loader when _isLoading is true
                  : Text(
                      'GO',
                      style: TextStyle(
                        fontFamily: 'Lato',
                        fontSize: 14,
                        color: const Color(0xffecf3e8),
                        fontWeight: FontWeight.w700,
                      ),
                    ),
            ),
          ),
        ],
      ),
    );
  }
}

const String _svg_tjcyqk =
    '<svg viewBox="0.0 0.0 39.6 39.7" ><path transform="translate(-1.0, -0.98)" d="M 11.12230205535889 38.09334564208984 C 10.54193687438965 37.76599884033203 9.978416442871094 37.40962219238281 9.433870315551758 37.02563858032227 C 9.65573787689209 36.95718002319336 9.881570816040039 36.90231323242188 10.11016464233398 36.861328125 C 10.46873378753662 36.79768371582031 10.8213586807251 36.70367050170898 11.16407108306885 36.58027648925781 C 11.12286949157715 37.08354949951172 11.10892295837402 37.58858871459961 11.12230205535889 38.09334564208984 L 11.12230205535889 38.09334564208984 Z M 29.05245780944824 38.84156799316406 C 23.82681655883789 41.23701858520508 17.81684494018555 41.23701858520508 12.59120273590088 38.84156799316406 C 12.50009346008301 37.85385513305664 12.52699279785156 36.85884475708008 12.6712703704834 35.87757110595703 C 12.91345691680908 35.74037551879883 13.13716602325439 35.60303497314453 13.34218502044678 35.46909332275391 L 13.34218502044678 36.50947952270508 C 13.34218502044678 36.90047073364258 13.6591272354126 37.21741485595703 14.05012130737305 37.21741485595703 C 14.44111633300781 37.21741485595703 14.75805950164795 36.90047073364258 14.75805950164795 36.50947952270508 L 14.75805950164795 33.43001937866211 C 15.11323070526123 33.87078475952148 15.41134643554688 34.29957962036133 15.68722820281982 34.70020294189453 C 16.5797233581543 35.99580001831055 17.42245292663574 37.2187614440918 19.46952629089355 37.2187614440918 L 22.17413330078125 37.2187614440918 C 24.22120666503906 37.2187614440918 25.06386375427246 35.99580001831055 25.95642852783203 34.70020294189453 C 26.23231887817383 34.29957962036133 26.53035736083984 33.87078475952148 26.88559722900391 33.43001937866211 L 26.88559722900391 36.50947952270508 C 26.88559722900391 36.90047073364258 27.20254325866699 37.21741485595703 27.59353828430176 37.21741485595703 C 27.98452949523926 37.21741485595703 28.30147361755371 36.90047073364258 28.30147361755371 36.50947952270508 L 28.30147361755371 35.46930694580078 C 28.50656318664551 35.60338973999023 28.73027420043945 35.74073028564453 28.97245979309082 35.87778854370117 C 29.11673736572266 36.85906219482422 29.14356994628906 37.85392379760742 29.05245780944824 38.84156799316406 L 29.05245780944824 38.84156799316406 Z M 12.45103168487549 22.33621025085449 C 12.16764354705811 20.95424270629883 12.01416397094727 19.5488452911377 11.99257183074951 18.13834953308105 C 13.61700534820557 19.73998832702637 16.91882514953613 20.8159122467041 20.82182884216309 20.8159122467041 C 24.72483253479004 20.8159122467041 28.02665328979492 19.73998832702637 29.65108871459961 18.13834953308105 C 29.62949562072754 19.5488452911377 29.47601318359375 20.95424270629883 29.192626953125 22.33621025085449 C 29.01833534240723 23.11331176757812 28.92474365234375 23.90634346008301 28.9133472442627 24.70263481140137 C 28.9133472442627 25.01115036010742 28.92906188964844 25.30947685241699 28.94251441955566 25.60872268676758 C 28.92254829406738 25.71066665649414 28.92594718933105 25.81572151184082 28.95235252380371 25.91611099243164 L 28.95341873168945 25.94916915893555 C 29.04537391662598 27.83794593811035 29.11800956726074 29.32985687255859 27.12481117248535 31.11973762512207 L 27.10774993896484 31.13651466369629 C 27.09946823120117 31.14437484741211 27.09167861938477 31.15180206298828 27.08375358581543 31.16001510620117 C 26.21036720275879 31.97527694702148 25.44013214111328 32.89439392089844 24.79017448425293 33.89690399169922 C 23.87969589233398 35.21869659423828 23.42477607727051 35.80288696289062 22.17413330078125 35.80288696289062 L 19.46952629089355 35.80288696289062 C 18.2188835144043 35.80288696289062 17.7640323638916 35.21869659423828 16.85348510742188 33.89690399169922 C 16.2035961151123 32.89439392089844 15.43336009979248 31.97527694702148 14.55997943878174 31.16008758544922 C 14.55205154418945 31.15180206298828 14.54419136047363 31.14437484741211 14.53583812713623 31.13651466369629 L 14.51955509185791 31.12044525146484 C 12.52564811706543 29.32985687255859 12.59828281402588 27.83794593811035 12.69024562835693 25.94916915893555 L 12.69130420684814 25.91617965698242 C 12.71771144866943 25.81572151184082 12.72111034393311 25.71066665649414 12.70114612579346 25.60879135131836 C 12.71459579467773 25.30947685241699 12.73031234741211 25.01115036010742 12.73031234741211 24.70263481140137 C 12.71891403198242 23.90634346008301 12.62532520294189 23.11331176757812 12.45103168487549 22.33621025085449 Z M 10.49797248840332 15.15035820007324 L 10.70327472686768 14.94505500793457 C 10.83608341217041 14.81231594085693 10.91069889068604 14.63221645355225 10.91069889068604 14.44447135925293 C 10.91997337341309 9.793744087219238 14.15659618377686 5.772748470306396 18.6980152130127 4.770124912261963 L 18.6980152130127 9.488905906677246 C 18.6980152130127 9.879900932312012 19.01495933532715 10.19684314727783 19.40595245361328 10.19684314727783 C 19.79694747924805 10.19684314727783 20.1138916015625 9.879900932312012 20.1138916015625 9.488905906677246 L 20.1138916015625 3.825403213500977 L 21.52976608276367 3.825403213500977 L 21.52976608276367 9.488905906677246 C 21.52976608276367 9.879900932312012 21.84671211242676 10.19684314727783 22.23770713806152 10.19684314727783 C 22.62869834899902 10.19684314727783 22.94564628601074 9.879900932312012 22.94564628601074 9.488905906677246 L 22.94564628601074 4.770124912261963 C 27.4870662689209 5.772748470306396 30.72368621826172 9.793744087219238 30.73296165466309 14.44447135925293 C 30.73296165466309 14.63221645355225 30.80757331848145 14.81231594085693 30.94038391113281 14.94498348236084 L 31.15057182312012 15.15240955352783 L 10.49797248840332 15.15035820007324 Z M 11.30360507965088 24.39956474304199 C 10.69810676574707 23.79399681091309 10.28219318389893 23.02503204345703 10.1066951751709 22.18683815002441 C 9.967231750488281 21.27578926086426 9.979335784912109 20.34789276123047 10.14258670806885 19.4408130645752 C 10.34880924224854 19.53850936889648 10.52798843383789 19.6850528717041 10.66469097137451 19.86770057678223 C 10.7434139251709 20.78639221191406 10.87544441223145 21.6997013092041 11.06000423431396 22.60303115844727 C 11.18828201293945 23.19451522827148 11.26976680755615 23.7952709197998 11.30360507965088 24.39956474304199 Z M 30.97896766662598 19.86763000488281 C 31.11566925048828 19.6850528717041 31.294921875 19.53850936889648 31.50107002258301 19.4408130645752 C 31.66297912597656 20.33890533447266 31.6765022277832 21.2574520111084 31.54114151000977 22.15986251831055 C 31.36875915527344 23.00790023803711 30.95114707946777 23.78656387329102 30.34012413024902 24.39935111999512 C 30.37396430969238 23.79512786865234 30.45537376403809 23.1944408416748 30.58365440368652 22.60303115844727 C 30.76828575134277 21.69963264465332 30.90024375915527 20.78632354736328 30.97896766662598 19.86763000488281 L 30.97896766662598 19.86763000488281 Z M 29.11093139648438 16.56828498840332 C 28.30473518371582 18.01977157592773 25.12828636169434 19.4000358581543 20.82182884216309 19.4000358581543 C 16.51537322998047 19.4000358581543 13.33892822265625 18.01977157592773 12.53272914886475 16.56828498840332 L 29.11093139648438 16.56828498840332 Z M 30.5213565826416 38.09328079223633 C 30.5346622467041 37.58852005004883 30.52079010009766 37.08340072631836 30.47958946228027 36.58013153076172 C 30.82364463806152 36.70394897460938 31.1776123046875 36.79824829101562 31.53767013549805 36.86203002929688 C 31.76491928100586 36.90280532836914 31.98940277099609 36.95739364624023 32.20993041992188 37.02556610107422 C 31.66531372070312 37.40954971313477 31.10179328918457 37.7658576965332 30.5213565826416 38.09328079223633 Z M 33.55019378662109 35.9962272644043 C 32.97797012329102 35.75764846801758 32.38217163085938 35.58002471923828 31.77277755737305 35.46619415283203 C 31.13875007629395 35.34244155883789 30.52801132202148 35.12015533447266 29.96279525756836 34.80738067626953 C 29.37895584106445 34.49837112426758 28.82308387756348 34.13909149169922 28.30147361755371 33.73372650146484 L 28.30147361755371 31.95298004150391 C 30.44815444946289 29.89641952514648 30.46521949768066 27.98966217041016 30.37969589233398 26.14477348327637 C 31.72570037841797 25.29425811767578 32.65501022338867 23.92000961303711 32.94314193725586 22.35412216186523 C 33.23077774047852 19.94274139404297 32.96940231323242 18.52962684631348 32.16469192504883 18.15491485595703 C 31.82084655761719 17.99378776550293 31.42765998840332 17.97304725646973 31.06873321533203 18.09721755981445 C 31.07906913757324 17.63224411010742 31.08183097839355 17.12691879272461 31.07382965087891 16.56828498840332 L 31.15050506591797 16.56828498840332 C 31.72308349609375 16.56694030761719 32.23846054077148 16.22090148925781 32.45650482177734 15.69150352478027 C 32.67447662353516 15.16203784942627 32.55228424072266 14.55342292785645 32.14671325683594 14.14926052093506 L 32.14465713500977 14.14720821380615 C 31.99153137207031 8.794066429138184 28.11472320556641 4.277612209320068 22.84660148620605 3.315029382705688 C 22.6363468170166 2.770576000213623 22.1133918762207 2.411013841629028 21.52976608276367 2.409527063369751 L 20.1138916015625 2.409527063369751 C 19.53019714355469 2.410992860794067 19.00731468200684 2.770554304122925 18.7969856262207 3.315014839172363 C 13.52893829345703 4.277597904205322 9.652128219604492 8.794066429138184 9.498930931091309 14.14713668823242 L 9.49687671661377 14.14926052093506 C 9.091300010681152 14.55342292785645 8.969109535217285 15.16203784942627 9.187154769897461 15.69150352478027 C 9.405198097229004 16.22090148925781 9.920578002929688 16.56694030761719 10.49315738677979 16.56828498840332 L 10.56982803344727 16.56828498840332 C 10.56182861328125 17.12727355957031 10.5645170211792 17.6328125 10.57485389709473 18.09785842895508 C 10.21578788757324 17.97332954406738 9.822245597839355 17.99407196044922 9.478259086608887 18.15562438964844 C 8.674182891845703 18.52962684631348 8.412882804870605 19.94274139404297 8.703916549682617 22.37769317626953 C 8.996719360351562 23.93459129333496 9.924188613891602 25.29935455322266 11.26395988464355 26.14477348327637 C 11.17851257324219 27.98944664001465 11.1955738067627 29.89592361450195 13.34218502044678 31.95298004150391 L 13.34218502044678 33.73372650146484 C 12.81923007965088 34.13987350463867 12.26201248168945 34.49978637695312 11.67668914794922 34.8094367980957 C 11.1140193939209 35.12064743041992 10.50611400604248 35.34201812744141 9.875058174133301 35.46555709838867 C 9.264177322387695 35.57939147949219 8.666961669921875 35.75722122192383 8.093390464782715 35.9962272644043 C 1.702722787857056 30.64322280883789 -0.6557931303977966 21.86380195617676 2.192042350769043 14.02891159057617 C 5.039877891540527 6.194071769714355 12.48543643951416 0.9782630205154419 20.8217601776123 0.9782630205154419 C 29.15815353393555 0.9782630205154419 36.60367584228516 6.194071769714355 39.45156860351562 14.02891159057617 C 42.29939270019531 21.86380195617676 39.94089126586914 30.64322280883789 33.55019378662109 35.9962272644043 L 33.55019378662109 35.9962272644043 Z" fill="#0f8168" stroke="none" stroke-width="1" stroke-miterlimit="4" stroke-linecap="butt" /></svg>';
const String _svg_raf7y =
    '<svg viewBox="0.0 0.0 39.6 39.6" ><path transform="translate(-1.0, -1.0)" d="M 20.8232536315918 40.64650344848633 C 9.830358505249023 40.64650344848633 1 31.81615257263184 1 20.8232536315918 C 1 9.830357551574707 9.830358505249023 1 20.8232536315918 1 C 24.60769653320312 1 28.39213180541992 2.081268310546875 31.63593673706055 4.243805885314941 C 32.53699493408203 4.784438610076904 32.71720504760742 5.865707874298096 32.17657089233398 6.766764163970947 C 31.63593673706055 7.667822360992432 30.55466461181641 7.84803295135498 29.65360641479492 7.307399272918701 C 26.950439453125 5.505284786224365 23.88684844970703 4.604228019714355 20.8232536315918 4.604228019714355 C 11.8126802444458 4.604228019714355 4.604227542877197 11.81268119812012 4.604227542877197 20.8232536315918 C 4.604227542877197 29.83382606506348 11.8126802444458 37.04228210449219 20.8232536315918 37.04228210449219 C 29.83382415771484 37.04228210449219 37.04227828979492 29.83382606506348 37.04227828979492 20.8232536315918 C 37.04227828979492 17.57944679260254 36.14121627807617 14.33564186096191 34.15889358520508 11.63247108459473 C 33.61826324462891 10.73141288757324 33.79846954345703 9.650146484375 34.69952774047852 9.109512329101562 C 35.60058212280273 8.568879127502441 36.68185043334961 8.749090194702148 37.22248458862305 9.650146484375 C 39.3850212097168 12.89395332336426 40.64649963378906 16.85860061645508 40.64649963378906 20.8232536315918 C 40.64649963378906 31.81615257263184 31.8161506652832 40.64650344848633 20.8232536315918 40.64650344848633 Z M 29.83382415771484 15.41691112518311 L 26.22959136962891 15.41691112518311 L 26.22959136962891 11.81268119812012 C 26.22959136962891 10.73141288757324 25.50875091552734 10.01056957244873 24.42748260498047 10.01056957244873 L 17.21902465820312 10.01056957244873 C 16.13775634765625 10.01056957244873 15.41690921783447 10.73141288757324 15.41690921783447 11.81268119812012 L 15.41690921783447 15.41691112518311 L 11.8126802444458 15.41691112518311 C 10.73141193389893 15.41691112518311 10.01056861877441 16.13775444030762 10.01056861877441 17.21902465820312 L 10.01056861877441 24.42748260498047 C 10.01056861877441 25.50875091552734 10.73141193389893 26.22959327697754 11.8126802444458 26.22959327697754 L 15.41690921783447 26.22959327697754 L 15.41690921783447 29.83382606506348 C 15.41690921783447 30.91509246826172 16.13775634765625 31.63593864440918 17.21902465820312 31.63593864440918 L 24.42748260498047 31.63593864440918 C 25.50875091552734 31.63593864440918 26.22959136962891 30.91509246826172 26.22959136962891 29.83382606506348 L 26.22959136962891 26.22959327697754 L 29.83382415771484 26.22959327697754 C 30.91508865356445 26.22959327697754 31.63593673706055 25.50875091552734 31.63593673706055 24.42748260498047 L 31.63593673706055 17.21902465820312 C 31.63593673706055 16.13775444030762 30.91508865356445 15.41691112518311 29.83382415771484 15.41691112518311 Z" fill="#0f8168" stroke="none" stroke-width="1" stroke-miterlimit="4" stroke-linecap="butt" /></svg>';
