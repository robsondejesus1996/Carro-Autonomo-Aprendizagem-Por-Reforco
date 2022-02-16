


<!-- Visualizador online: https://stackedit.io/ -->
 ![Logo da UDESC Alto Vale](http://www1.udesc.br/imagens/id_submenu/2019/marca_alto_vale_horizontal_assinatura_rgb_01.jpg)

---
# Carro Autônomo Aprendizagem Por Reforco

---

Trabalho realizado para a disciplina de Inteligência Computacional do [Centro de Educação Superior do Alto Vale do Itajaí (CEAVI/UDESC)](https://www.udesc.br/ceavi)<br> O objetivo do trabalho é implementar alguma técnica de Inteligência Computacional para demonstrar sua aplicação. A técnica de Inteligência Computacional que será utilizada no trabalho será alguns algoritmos de sistema de aprendizagem por reforço, abaixo irie explicar os algoritmos que serão implementados.


---

# Sumário 
* [Equipe](#equipe)
* [Introdução](#introdução)
* [ProblemaTrabalho](#problemaTrabalho)
* [Técnica](#tecnica)
* [Niveis de Resultados](#resultados)
* [Vídeo](#Vídeo)
* [Instrução de Uso](#instrucao)
* [Referencias](#Referencias)

---




## [Equipe](#equipe)
 - [**Robson de Jesus**](mailto:robson.jesus@edu.udesc.br) - [robsondejesus1996](https://github.com/robsondejesus1996)



---


## [Introdução](#introdução)

<p>

O aprendizado por reforço, ou reinforcement learning, possui aplicações e usos fantásticos. Muitas vezes assustadores ao terem se tornado melhores que humanos eleitos mundialmente como melhores em determinada área. (Deeplearningbook, 2018)
</p>


<p>
O funcionanmento desse trabalho vai ser meio parecido com o cenário de ensinar novos truques a um cachorro. Pense que o cão não entende nossa língua, então não podemos dizer a ele o que fazer. Em vez disso, seguimos uma estratégia diferente. Imitamos uma situação e esse cão tenta reagir de muitas maneiras diferentes. Se a resposta do cão for a desejada, nós o recompensamos com uma comida diferente. Agora da proximá vez que o cão for exposto a mesma situação, ele executará uma ação semelhante com ainda mais entusiasmo na expectativa de mais comida. Então basicamente o trabalho que será desenvolvido terá esse objetivo, é como aprender "o que fazer" com experiências positivas. (Petz, 2021)
</p>



---

## [ProblemaTrabalho](#problemaTrabalho)

O do trabalho que será desenvolvido é fazer a implementação passo a posso de uma inteligência artificial para controlar um carro autônomo virtual. Então basicamente vai ser criado um cenário para esse carro e ele tem que aprender com o ambiente. Logo será desenhado obstáculos e o carro tem que descobrir melhores rotas para chegar no seu destino. Então basicamente como já foi dito o tipo de de aprendizagem usado em no sistema desenvolvido será um multi-agente no qual os agentes devem interagir no ambiente e aprenderem por conta própria, ganhando recompensas positivas quando executam ações corretas e recompensas negativas quando executam ações que não levem para o objetivo. O interessante é que a proposta do trabalho é desenvolver uma técnica de inteligência artificial que  aprende sem nenhum conhecimento prévio, adaptando-se ao ambiente e encontrando as soluções sozinho. 

<h1> Multi-agente </h1>
<p>Basicamente o meu agente do sistema seria o carro, o agente tem uma existência própria, independente da existencia de outros agentes, basicamente o trabalho foi pensado eu envolver diversos agentes(carros), mas nesse primeira versão foi colocado somente um agente<p>

![carro](https://user-images.githubusercontent.com/31260719/154281850-f1f6806b-bd0b-416e-8e66-1a1284da0209.png)


<h1> Obstáculos do ambiente </h1>

<p>Basicamente o sistema foi pensado como os obstáculos que o carro consegui-se passar, por exemplo se eu coloca-se obstáculos como parede ou outros carros o agente iria ficar travado, então foi pensado em areia de praia por exemplo. Pois quando um carro passa na areia de praia ele diminui a velocidade mas o veículo não para totalmente<p>

![areia](https://user-images.githubusercontent.com/31260719/154283046-93dc415b-1eae-48be-a031-513ae1425eda.png)


<h1> Caminho para se chegar no objetivo </h1>

<p>Como foi descrito o caminho são as extremidades, basicamente no canto superior esquerdo do ambiente é um ponto de objetivo e o canto inferior direito é outro objetivo<p>

![objetivos](https://user-images.githubusercontent.com/31260719/154283950-2951dba6-0938-4b0a-b410-a12712ddedc9.png)



<h1> Recompensas positivas e negativas </h1>
<p>Basicamente no sistema terá um geração dos graficos de recompensa, os valores de score vão ser inicializados com a média das
recompensas com relação ao tempo. Vai haver uma adição dos valores das recompensas(média das 1000 últimas recompensas)<p>

<p>

<b>Recompensas Negativas: </b> Caso o carro está na areia, vai dimuir a velocidade de 6 para 1. Caso contrário, se não estiver na areia. <br>  

last_reward = -1 # ganha uma recompensa negativa <br>  

last_reward = -0.2 # ganha uma recompensa negativa pequena <br>  

<b>Recompensas Positivas: </b> Caso o carro esteja chegando no objetivo, quando o carro chega no objetivo ele muda para o canto inferior direito e vice-versa. <br>  

last_reward = 0.1 # ganha uma pequena recompensa positiva <br>  

</p>


---

## [Técnica](#tecnica)


<h1> Técnica, linguagem, algoritmo e biblioteca usado abaixo:</h1>
- Aprendizagem por Reforço
- Python 
- Deep Learning
- Biblioteca PyTorch 



<h1> Aplicação da Técnica no problema </h1>

![equacao](https://user-images.githubusercontent.com/31260719/154358635-81f1d36d-e188-4720-aa1a-73f6b405f32b.png)


<p>A técnica de sistema desenvolvido será de aprendizagem por reforço com intuição Deep Q-Learning, basicamente serão 5 entradas que tem na rede neural, então o carro tem 3 sensores (um na esquerda, um na direita, e outro no meio) esse sensores serão utilizados como entrada da rede neural. E além disso é utilizado a direção que nada mas é que a rodação do carro(direita, esquerda) 5 entradas = 3 sensores + duas direções. Então basicamente o algoritmo de Deep Q-Learning vai resolver o problema do meu agente no ambiente. <p>


<h1> Estados e Ações </h1>
<p>Existe 3 ações no sistema, o carro ir para frente, para a esquerda, ou para a direita. Essa ações equivale as saídas da rede neural(camada de saída)<p>

<h1> Calculo da Recompensa </h1>

![experiencia](https://user-images.githubusercontent.com/31260719/154359915-316ea1cf-d03c-4d1b-bde2-ff5cd532485a.png)

<p>Basicamente na inicialização o valor médio das recompensas tem uma relação ao tempo de execução do ambiente, basicamente na experiência de replay vai ser armazenado as recompensas que tem no caminho, podendo fazer a média das recompensas em um determinado periodo. Como já foi dito se o carro for para areia ele acaba diminuindo a velocidade e vai ganhar uma recompensa negativa. <p>


<h1>taxa de aprendizagem e fator de desconto </h1>
<p>A taxa de fator de desconto é 0,9 que é o valor do gamma  (y = 0.9) <p>




## [Niveis de Resultados](#resultados)

<p> Para a validação da aplicação será utilizado 4 níveis de dificuldade no sistema:</p>

<ul>
    <b>Niveis:</b><br>
    <li> 1)	Nível 1 não será colocado nenhum obstáculo.</li>
    <li> 2)	Nível 2 desenhar uma estrada e esse carro deve se manter dentro da estrada.</li>
    <li> 3)	Nível 3 desenhar objetos no mapa.</li>
    <li> 4)	Nível 4 desenhar estrada mais complexa.</li>  
</ul>


<h1>Nível 1</h1>

<p>Primeiro teste sem utilizar a inteligência artificial. Mudança do valor da temperatura para zero. Nesse tipo de simulação ele esta sem nenhum objetivo definido então ele fica andando para todos os lados como se fosse um inseto. Basicamente o objetivo principal é ir de uma lado para o outro. </p>

<p>Para realizar as viagens de uma ponta da extremidade para outra tem que mudar o valor do parâmetro para <b> 7.</b> Assim podemos notar na simulação que ele tenta chegar de uma extremidade para outra quando é mudado esse parâmetro</p>

![temperatura_7](https://user-images.githubusercontent.com/31260719/154275972-f41c9932-5a73-480f-9446-362b52c489cf.png)


![nivel_2](https://user-images.githubusercontent.com/31260719/154275726-9179eda8-ac08-40da-aaa1-aaba0b0faf68.png)


<p>Então basicamente ele nessa situação já está no nível 2. Nesse caso como não a obstáculos de areia na pista as únicas punições é quando ele chega nas bordas e também ele tem a punição quando ele está longe do objetivo. <br>
Realizando outro teste caso você queira deixar o carro com menos movimentos. Por que atualmente com o parâmetro 7 ele parece que não está muito confiante nos movimentos que ele esta tomando. Altere o valor da temperatura para 100. 
</p>

![parametro_temperatura](https://user-images.githubusercontent.com/31260719/154275453-5eaa47fa-5fd9-44d2-ab38-40b5fd5b84d7.png)


<p>Basicamente agora o carro está mas confiante nesse movimentos, isso acontece por que o parâmetro da temperatura ele é um parâmetro para a função softmax retornar os dados com mais confiabilidade ou seja a inteligência artificial será mais confiança sobre a ação a ser tomada. É importante também dizer se esse parâmetro for muito aumentado pode atrapalhar o carro de explorar o ambiente. Basicamente salve este modole e passe para o segundo nível. </p>


<h1>Nível 2</h1>
<p>Abaixo temos o resultado da função <b>save<b> que faz um plot na variável scores. Lembrando como explicado no vídeo está variavel score que vai calcular os valores da recomepnsa, que é a media das 1000 últimas recompensas.</p>

![resultado](https://user-images.githubusercontent.com/31260719/154276735-2f02b8d8-4126-4f42-91fb-03845312bd57.png)

<p>Então nota-se no gráfico que ele começa com o valor de recompensa negativo e depois ele vai chegando numa recompensa positiva de 0.10 que é o maior valor de recompensa positiva definido.</p>

<p>Ainda no nível 2, Execute o mapa e aperte em load para carregar o modelo anterior, seja que quando você faz esse carregamento ele já consegue fazer esta roda do nível 1. Agora o objetivo no nível 2 e verificar se ele vai conseguir andar dentro de uma estrada. Desenho a estrada. Facilmente podemos notar que esse agente conseguiu passar para o nível 2 dentro dessa estrada.</p>

![nivel_2_estrada](https://user-images.githubusercontent.com/31260719/154277242-9d68cd05-d81c-4275-a60c-e91395ef0735.png)





<h1>Nível 3</h1>
<p>Nesse teste vai ser colocado alguns obstáculos, no meio do mapa. Pode-se notar que ele também consegue se adaptar a esse novo ambiente. Então ele já passou para o nível 3.</p>

![nivel_3](https://user-images.githubusercontent.com/31260719/154277583-76ff7e8e-1f7c-42e2-9da0-1c2586a3c1dd.png)


<h1>Nível 4</h1>

![nivel_4](https://user-images.githubusercontent.com/31260719/154278019-d529c371-1c2c-4d50-b62d-9b9784f582a9.png)

<p>4)	Para o agente conseguir passar no nível 4 tem que mudar a estrutura da rede neural 
Pode-se notar que o agente não se adapta muito bem ao ambiente colocado na pista
Então basicamente para o agente conseguir se adaptar a esse tipo de ambiente deve-se mudar a estrutura da rede neural. Então tem que ter uma melhoria do código fonte tem que mudar a estrutura da rede. Temos uma rede neural bastante simples que tem somente uma camada oculta. Então se buscar na teoria de deep larning é considerado uma deep larning quando tem pelo menos duas camadas ocultas. Então basicamente pode-se adicionar mas camadas ocultas, adicionar mais a quantidade de neurônios e verificar como vai ficar esse desempenho. 
Coisas para se alterar:
</p>


<ul>
    <b>Alterações possíveis:</b><br>
     1 - Adicionar mais camadas ocultas:<br>

![camadas](https://user-images.githubusercontent.com/31260719/154278826-19497c85-1dc2-408a-8379-50ca349f1e6d.png)

     2 - Pode aumentar a quantidade de neurônios  
     3 - Pode-se alterar o parâmetro da temperatura  

![temp](https://user-images.githubusercontent.com/31260719/154279081-421a0ac4-596a-4169-934d-95c98b799de4.png)


     4 - Pode usar outras funções de erro
     5 - Pode utilizar outros tipos de utilizadores (é utilizado o Adan, mas também tem o rms)

![adam](https://user-images.githubusercontent.com/31260719/154279308-d53460ab-27f6-4312-92e6-8d3cb2df8b1b.png)


     6 - Com relação ao mapa pode dentar diminuir os valores das recompensas.

![recompensas](https://user-images.githubusercontent.com/31260719/154279715-faf89e2b-aab1-4f17-b2f2-d4d5bc0d5fb0.png)

</ul>


## [Vídeo](#Vídeo)

>Figura : Video Apresentação trabalho (clique para abrir no YouTube)

[![capa](https://user-images.githubusercontent.com/31260719/154288443-0e2c6179-bff8-40ac-a954-6bd8f8f48384.png)](https://www.youtube.com/watch?v=bi5DT2NcGME&ab_channel=robsonjesus)



## [Instrução de Uso](#instrucao)

1) Instale o Anaconda navigator (anaconda3)<br>
2) No ambiente anaconta basicamente entre na opção para criar o ambiente de desenvolvimento em (Environments) nessa opçao tem uma função de criação chamada "create". Basicamente defina um nome para o projeto a versão do python deve ser a 3.6.13. <br>

![criação projeto](https://user-images.githubusercontent.com/31260719/154360909-43ae02c7-bf37-4d55-8fb1-e6837980aec7.png)

3)Com o ambiente criado entre na opção de home e certifique-se que está na opção de applications on = nome do projeto que você criou. Basicamente nesta etapa tem que se instalar no ambiente o Spyder 5.0.5. <br>

![instalacao](https://user-images.githubusercontent.com/31260719/154361459-d4b37032-a148-40a8-b38d-c590ec3492f7.png)

4) Execute o Spyder no ambiente de desenvolvimento. <br>

5) Temos que instalar mais algumas dependecias do projeto, execute o Anaconda Prompt e insira os comandos abaixo: <br>
activate carro_autonomo <br>
conda install -c peterjc123 pytorch-cpu <br>
conda install -c conda-forge kivy <br>
pip install matplotlib <br>

6) com o Spyder rodando no canto superior esquerdo abra o projeto no caminho do arquivo, basicamente pode abrir o arquivo map.py e rodar a aplicação que vai funcionar. 

![rodando](https://user-images.githubusercontent.com/31260719/154362695-5e17bc8b-3440-43af-ae97-7299efa41808.png)


## [Referencias](#Referencias)
O Deeplearningbook [Site de noticias].disponivel em: <br>
<https://www.deeplearningbook.com.br/aplicacoes-da-aprendizagem-por-reforco-no-mundo-real/#:~:text=O%20Aprendizado%20por%20Refor%C3%A7o%20(ou,de%20maximizar%20as%20recompensas%20totais.&text=Isso%20%C3%A9%20o%20Aprendizado%20por%20Refor%C3%A7o.><br>

O petz [Site de noticias].disponivel em: <br>
<https://www.petz.com.br/blog/bem-estar/caes-bem-estar/ensinar-reforco-positivo/>